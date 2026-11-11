import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.aiter_backend import AiterIndicesUpdaterPrefill
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(torch.cuda.is_available(), "Test requires a GPU")
class TestAiterPrefillIndicesUpdater(unittest.TestCase):
    def test_kv_indices_tail_is_padded_without_host_sync(self):
        seq_lens = [300, 1, 4096]
        bs, total = len(seq_lens), sum(seq_lens)
        req_to_token = torch.randperm((bs + 1) * 4104, dtype=torch.int32).cuda()
        req_to_token = req_to_token.reshape(bs + 1, 4104)
        updater = SimpleNamespace(
            kv_indptr=torch.zeros(bs + 2, dtype=torch.int32, device="cuda"),
            qo_indptr=torch.zeros(bs + 2, dtype=torch.int32, device="cuda"),
            req_to_token=req_to_token,
            kv_index_translator=None,
        )
        args = (
            torch.arange(1, bs + 1, dtype=torch.int64, device="cuda"),
            torch.tensor(seq_lens, dtype=torch.int64, device="cuda"),
            total,
            torch.zeros(bs, dtype=torch.int64, device="cuda"),
            None,
            None,
        )
        AiterIndicesUpdaterPrefill.update_single_wrapper(updater, *args, plan=None)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")  # any host sync raises
        try:
            AiterIndicesUpdaterPrefill.update_single_wrapper(updater, *args, plan=None)
        finally:
            torch.cuda.set_sync_debug_mode(0)

        kv_indices = updater.kv_indices.cpu()
        expected = torch.cat(
            [req_to_token[i + 1, :n].cpu() for i, n in enumerate(seq_lens)]
        )
        self.assertTrue(torch.equal(kv_indices[:total], expected))
        self.assertTrue((kv_indices[total:] == expected[0]).all())


if __name__ == "__main__":
    unittest.main()
