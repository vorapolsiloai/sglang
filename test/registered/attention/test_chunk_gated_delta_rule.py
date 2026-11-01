import itertools
import unittest

import torch

from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)
from sglang.srt.environ import envs
from sglang.srt.utils import get_device, is_hip
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cuda_ci,
    register_xpu_ci,
)

register_cuda_ci(est_time=7, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=11, stage="stage-b", runner_config="1-gpu-large-amd")
register_xpu_ci(est_time=900, suite="stage-b-test-1-gpu-xpu")


@unittest.skipIf(
    not (torch.cuda.is_available() or torch.xpu.is_available()),
    "Test requires CUDA or XPU",
)
class TestChunkGatedDeltaRule(unittest.TestCase):
    """Test chunk_gated_delta_rule against token-by-token fused_recurrent reference."""

    ATOL = 2e-2
    RTOL = 1e-2

    def _run_reference(self, pool_init, cache_indices, q, k, v, g, beta):
        """Per-batch token-by-token reference using fused_recurrent_gated_delta_rule.

        initial_state shape: [N, H, V, K] (native layout on this branch).
        """
        B = cache_indices.shape[0]
        T_per_seq = q.shape[1] // B
        pool = pool_init.clone()
        h_cur = pool[cache_indices].contiguous().clone()

        o_list = []
        for b in range(B):
            sl = slice(b * T_per_seq, (b + 1) * T_per_seq)
            o_b, h_b = fused_recurrent_gated_delta_rule(
                q=q[0, sl].unsqueeze(0),
                k=k[0, sl].unsqueeze(0),
                v=v[0, sl].unsqueeze(0),
                g=g[0, sl].unsqueeze(0),
                beta=beta[0, sl].unsqueeze(0),
                initial_state=h_cur[b : b + 1],
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            o_list.append(o_b)
            h_cur[b] = h_b[0]

        pool[cache_indices] = h_cur
        return torch.cat(o_list, dim=1), pool

    def _run_chunk(self, pool_init, cache_indices, q, k, v, g, beta, cu_seqlens):
        """Run chunk_gated_delta_rule with native [V, K] pool."""
        pool = pool_init.clone()
        o, _, _ = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=pool,
            initial_state_indices=cache_indices,
            cu_seqlens=cu_seqlens,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        return o, pool

    def _check_shape(
        self, B, T_per_seq, H, K, V, pool_size, sequential_indices=False, seed=42
    ):
        """Run correctness check for one (B, T_per_seq, H, K, V, pool_size) config."""
        device = get_device()
        dtype = torch.bfloat16
        T = B * T_per_seq

        torch.manual_seed(seed)

        if sequential_indices:
            cache_indices = torch.arange(B, dtype=torch.int32, device=device)
        else:
            perm = torch.randperm(pool_size, device=device)[:B]
            cache_indices = perm.to(torch.int32)

        pool_init = (
            torch.randn(pool_size, H, V, K, dtype=torch.float32, device=device) * 0.1
        )
        cu_seqlens = torch.zeros(B + 1, dtype=torch.long, device=device)
        cu_seqlens[1:] = (
            torch.arange(1, B + 1, dtype=torch.long, device=device) * T_per_seq
        )

        q = torch.randn(1, T, H, K, dtype=dtype, device=device)
        k = torch.randn(1, T, H, K, dtype=dtype, device=device)
        v = torch.randn(1, T, H, V, dtype=dtype, device=device)
        g = torch.nn.functional.logsigmoid(
            torch.randn(1, T, H, dtype=dtype, device=device)
        )
        beta = torch.sigmoid(torch.randn(1, T, H, dtype=dtype, device=device))

        o_ref, pool_ref = self._run_reference(
            pool_init, cache_indices, q, k, v, g, beta
        )
        o_new, pool_new = self._run_chunk(
            pool_init, cache_indices, q, k, v, g, beta, cu_seqlens
        )

        self.assertTrue(
            torch.allclose(
                o_ref.float(), o_new.float(), atol=self.ATOL, rtol=self.RTOL
            ),
            f"Output mismatch: max_diff="
            f"{(o_ref.float() - o_new.float()).abs().max().item():.2e}",
        )

        ref_slots = pool_ref[cache_indices].contiguous()
        new_slots = pool_new[cache_indices].contiguous()
        self.assertTrue(
            torch.allclose(
                ref_slots.float(), new_slots.float(), atol=self.ATOL, rtol=self.RTOL
            ),
            f"State mismatch: max_diff="
            f"{(ref_slots.float() - new_slots.float()).abs().max().item():.2e}",
        )

    # ------------------------------------------------------------------
    # Production-style configs (Qwen3-Next)
    # ------------------------------------------------------------------
    def test_production_nt1(self):
        self._check_shape(B=4, T_per_seq=64, H=16, K=128, V=128, pool_size=32)

    def test_production_nt2(self):
        self._check_shape(B=4, T_per_seq=128, H=16, K=128, V=128, pool_size=32)

    def test_production_nt4(self):
        self._check_shape(B=4, T_per_seq=256, H=16, K=128, V=128, pool_size=32)

    # ------------------------------------------------------------------
    # Batch size sweep
    # ------------------------------------------------------------------
    def test_batch_1(self):
        self._check_shape(B=1, T_per_seq=128, H=16, K=128, V=128, pool_size=32)

    def test_batch_2(self):
        self._check_shape(B=2, T_per_seq=128, H=16, K=128, V=128, pool_size=32)

    def test_batch_8(self):
        self._check_shape(B=8, T_per_seq=128, H=16, K=128, V=128, pool_size=64)

    def test_batch_16(self):
        self._check_shape(B=16, T_per_seq=64, H=16, K=128, V=128, pool_size=128)

    def test_batch_32(self):
        self._check_shape(B=32, T_per_seq=32, H=16, K=128, V=128, pool_size=256)

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "The read-only initial-state path is not implemented by the XPU chunk kernel",
    )
    def test_read_only_initial_state_supports_duplicate_indices(self):
        """MIS item branches may share one query-end state without updating it."""
        device = get_device()
        dtype = torch.bfloat16
        batch_size, tokens_per_item = 3, 65
        num_heads, key_dim, value_dim = 4, 32, 32
        total_tokens = batch_size * tokens_per_item

        torch.manual_seed(1234)
        pool_init = torch.randn(
            4,
            num_heads,
            value_dim,
            key_dim,
            dtype=torch.float32,
            device=device,
        )
        duplicate_indices = torch.tensor([2, 2, 2], dtype=torch.int32, device=device)
        cu_seqlens = torch.arange(
            0,
            total_tokens + 1,
            tokens_per_item,
            dtype=torch.int32,
            device=device,
        )
        q = torch.randn(1, total_tokens, num_heads, key_dim, dtype=dtype, device=device)
        k = torch.randn_like(q)
        v = torch.randn(
            1, total_tokens, num_heads, value_dim, dtype=dtype, device=device
        )
        g = torch.nn.functional.logsigmoid(
            torch.randn(1, total_tokens, num_heads, dtype=dtype, device=device)
        )
        beta = torch.sigmoid(
            torch.randn(1, total_tokens, num_heads, dtype=dtype, device=device)
        )

        expected, _ = self._run_reference(
            pool_init, duplicate_indices, q, k, v, g, beta
        )
        actual_pool = pool_init.clone()
        actual, _, _ = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=actual_pool,
            initial_state_indices=duplicate_indices,
            cu_seqlens=cu_seqlens,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
            inplace_update=False,
        )

        torch.testing.assert_close(
            actual.float(), expected.float(), atol=self.ATOL, rtol=self.RTOL
        )
        self.assertTrue(torch.equal(actual_pool, pool_init))

        updating_pool = pool_init.clone()
        chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=updating_pool,
            initial_state_indices=torch.arange(
                batch_size, dtype=torch.int32, device=device
            ),
            cu_seqlens=cu_seqlens,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        self.assertFalse(
            torch.equal(updating_pool[:batch_size], pool_init[:batch_size])
        )

    # ------------------------------------------------------------------
    # Head count sweep
    # ------------------------------------------------------------------
    def test_heads_4(self):
        self._check_shape(B=4, T_per_seq=128, H=4, K=128, V=128, pool_size=32)

    def test_heads_8(self):
        self._check_shape(B=4, T_per_seq=128, H=8, K=128, V=128, pool_size=32)

    def test_heads_32(self):
        self._check_shape(B=4, T_per_seq=128, H=32, K=128, V=128, pool_size=32)

    def test_heads_64(self):
        self._check_shape(B=4, T_per_seq=128, H=64, K=128, V=128, pool_size=32)

    # ------------------------------------------------------------------
    # K != V  (exercises that [V,K] != [K,V] byte-order matters)
    # ------------------------------------------------------------------
    def test_dim_64x64(self):
        self._check_shape(B=4, T_per_seq=128, H=16, K=64, V=64, pool_size=32)

    def test_dim_k_lt_v(self):
        self._check_shape(B=4, T_per_seq=128, H=16, K=64, V=128, pool_size=32)

    def test_dim_k_gt_v(self):
        self._check_shape(B=4, T_per_seq=128, H=16, K=128, V=64, pool_size=32)

    @unittest.skipIf(
        is_hip(),
        "K=V=256 head dim exceeds the FLA chunk triton kernel's shared-memory "
        "budget on ROCm (out-of-resource at launch); smaller head dims pass.",
    )
    def test_dim_256x256(self):
        self._check_shape(B=4, T_per_seq=128, H=16, K=256, V=256, pool_size=32)

    # ------------------------------------------------------------------
    # Short sequences (T < chunk_size=64)
    # ------------------------------------------------------------------
    def test_seqlen_1(self):
        self._check_shape(B=4, T_per_seq=1, H=16, K=128, V=128, pool_size=32)

    def test_seqlen_7(self):
        self._check_shape(B=4, T_per_seq=7, H=16, K=128, V=128, pool_size=32)

    def test_seqlen_16(self):
        self._check_shape(B=4, T_per_seq=16, H=16, K=128, V=128, pool_size=32)

    def test_seqlen_32(self):
        self._check_shape(B=4, T_per_seq=32, H=16, K=128, V=128, pool_size=32)

    # ------------------------------------------------------------------
    # Multi-chunk and large pool
    # ------------------------------------------------------------------
    def test_multi_chunk_nt8(self):
        self._check_shape(B=4, T_per_seq=512, H=16, K=128, V=128, pool_size=32)

    def test_large_pool(self):
        self._check_shape(B=4, T_per_seq=128, H=16, K=128, V=128, pool_size=512)

    # ------------------------------------------------------------------
    # Long prompts (many chunks; regression test for cross-chunk errors)
    # ------------------------------------------------------------------
    def test_long_prompt(self):
        for B, T_per_seq in [(1, 1024), (1, 1536), (1, 2048), (2, 1024)]:
            with self.subTest(T_per_seq=T_per_seq):
                self._check_shape(
                    B=B, T_per_seq=T_per_seq, H=16, K=128, V=128, pool_size=32
                )

    # ------------------------------------------------------------------
    # Combined stress
    # ------------------------------------------------------------------
    def test_stress(self):
        self._check_shape(B=32, T_per_seq=128, H=32, K=128, V=128, pool_size=256)

    # ------------------------------------------------------------------
    # Sequential-index variants (pool_size == B, indices = 0..B-1)
    # ------------------------------------------------------------------
    def test_seq_idx_b4(self):
        self._check_shape(
            B=4,
            T_per_seq=128,
            H=16,
            K=128,
            V=128,
            pool_size=4,
            sequential_indices=True,
        )

    def test_seq_idx_b8(self):
        self._check_shape(
            B=8,
            T_per_seq=128,
            H=16,
            K=128,
            V=128,
            pool_size=8,
            sequential_indices=True,
        )

    def test_seq_idx_h32(self):
        self._check_shape(
            B=4,
            T_per_seq=128,
            H=32,
            K=128,
            V=128,
            pool_size=4,
            sequential_indices=True,
        )

    def test_seq_idx_h64(self):
        self._check_shape(
            B=4,
            T_per_seq=128,
            H=64,
            K=128,
            V=128,
            pool_size=4,
            sequential_indices=True,
        )

    def test_seq_idx_stress(self):
        self._check_shape(
            B=32,
            T_per_seq=128,
            H=32,
            K=128,
            V=128,
            pool_size=32,
            sequential_indices=True,
        )

    # ------------------------------------------------------------------
    # Padded rows (idle DP ranks under breakable CUDA graph prefill)
    # ------------------------------------------------------------------

    def test_padded_state_index_is_skipped(self):
        """Rows carrying state index -1 must be skipped, not addressed.

        Without the guard the sentinel reaches pointer arithmetic and
        addresses before the state pool.
        """
        device = get_device()
        dtype = torch.bfloat16
        B, T_per_seq, H, K, V, pool_size = 4, 64, 4, 128, 128, 8
        T = B * T_per_seq
        torch.manual_seed(0)

        pool_init = (
            torch.randn(pool_size, H, V, K, dtype=torch.float32, device=device) * 0.1
        )
        cu_seqlens = torch.zeros(B + 1, dtype=torch.long, device=device)
        cu_seqlens[1:] = (
            torch.arange(1, B + 1, dtype=torch.long, device=device) * T_per_seq
        )
        q = torch.randn(1, T, H, K, dtype=dtype, device=device)
        k = torch.randn(1, T, H, K, dtype=dtype, device=device)
        v = torch.randn(1, T, H, V, dtype=dtype, device=device)
        g = torch.nn.functional.logsigmoid(
            torch.randn(1, T, H, dtype=dtype, device=device)
        )
        beta = torch.sigmoid(torch.randn(1, T, H, dtype=dtype, device=device))

        # "all padded" is the idle-DP-rank case that faulted.
        for label, indices in (("mixed", [0, -1, 2, -1]), ("all_padded", [-1] * B)):
            with self.subTest(label):
                cache_indices = torch.tensor(indices, dtype=torch.int32, device=device)
                o, pool = self._run_chunk(
                    pool_init, cache_indices, q, k, v, g, beta, cu_seqlens
                )
                # Forces a sync: an out-of-bounds access surfaces here.
                self.assertTrue(torch.isfinite(o.float()).all())

                untouched = [s for s in range(pool_size) if s not in indices]
                self.assertTrue(
                    torch.equal(pool[untouched], pool_init[untouched]),
                    f"{label}: padded rows wrote into the state pool",
                )

    def test_rows_past_last_sequence_are_zero(self):
        """A padded batch's rows past cu_seqlens[-1] come out as zeros.

        The output is not zero-filled up front; the rows past the last sequence
        are cleared on their own, also when the allocator hands out blocks that
        still hold NaNs.
        """
        device = get_device()
        dtype = torch.bfloat16
        B, T_per_seq, pad, H, K, V, pool_size = 3, 64, 40, 4, 128, 128, 8
        T = B * T_per_seq
        torch.manual_seed(0)

        pool_init = (
            torch.randn(pool_size, H, V, K, dtype=torch.float32, device=device) * 0.1
        )
        cache_indices = torch.arange(B, dtype=torch.int32, device=device)
        cu_seqlens = torch.arange(0, T + 1, T_per_seq, dtype=torch.long, device=device)
        q = torch.randn(1, T + pad, H, K, dtype=dtype, device=device)
        k = torch.randn(1, T + pad, H, K, dtype=dtype, device=device)
        v = torch.randn(1, T + pad, H, V, dtype=dtype, device=device)
        g = torch.nn.functional.logsigmoid(
            torch.randn(1, T + pad, H, dtype=dtype, device=device)
        )
        beta = torch.sigmoid(torch.randn(1, T + pad, H, dtype=dtype, device=device))

        o_ref, _ = self._run_chunk(
            pool_init,
            cache_indices,
            q[:, :T],
            k[:, :T],
            v[:, :T],
            g[:, :T],
            beta[:, :T],
            cu_seqlens,
        )
        # Freed NaN blocks of the output's size, for the allocator to hand out next.
        nan_blocks = [torch.full_like(v, float("nan")) for _ in range(8)]
        del nan_blocks
        o, _ = self._run_chunk(pool_init, cache_indices, q, k, v, g, beta, cu_seqlens)

        self.assertTrue(torch.equal(o[:, T:], torch.zeros_like(o[:, T:])))
        self.assertTrue(
            torch.allclose(
                o[:, :T].float(), o_ref.float(), atol=self.ATOL, rtol=self.RTOL
            )
        )

    def test_output_buffer_is_written_in_place(self):
        """A contiguous buffer of v's shape and dtype, such as the real rows of a
        CUDA graph's static output, receives the output in place; any other
        buffer is left alone and a new tensor is returned. The output and the
        final states are the same either way."""
        device = get_device()
        dtype = torch.bfloat16
        B, T_per_seq, pad, H, K, V, pool_size = 3, 100, 40, 4, 128, 128, 8
        T = B * T_per_seq
        torch.manual_seed(0)

        pool_init = (
            torch.randn(pool_size, H, V, K, dtype=torch.float32, device=device) * 0.1
        )
        cache_indices = torch.arange(B, dtype=torch.int32, device=device)
        cu_seqlens = torch.arange(0, T + 1, T_per_seq, dtype=torch.long, device=device)
        q = torch.randn(1, T, H, K, dtype=dtype, device=device)
        k = torch.randn(1, T, H, K, dtype=dtype, device=device)
        v = torch.randn(1, T, H, V, dtype=dtype, device=device)
        g = torch.nn.functional.logsigmoid(
            torch.randn(1, T, H, dtype=dtype, device=device)
        )
        beta = torch.sigmoid(torch.randn(1, T, H, dtype=dtype, device=device))
        o_ref, pool_ref = self._run_chunk(
            pool_init, cache_indices, q, k, v, g, beta, cu_seqlens
        )

        def nans(*shape, dtype=dtype):
            return torch.full(shape, float("nan"), dtype=dtype, device=device)

        padded = nans(1, T + pad, H, V)
        for label, output, in_place in (
            ("contiguous", nans(1, T, H, V), True),
            ("real rows of a padded buffer", padded[:, :T], True),
            ("other dtype", nans(1, T, H, V, dtype=torch.float32), False),
            ("strided", nans(1, T, H, 2 * V)[..., :V], False),
        ):
            with self.subTest(label):
                pool = pool_init.clone()
                o, _, _ = chunk_gated_delta_rule(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    initial_state=pool,
                    initial_state_indices=cache_indices,
                    cu_seqlens=cu_seqlens,
                    head_first=False,
                    use_qk_l2norm_in_kernel=True,
                    output=output,
                )
                self.assertEqual(o.data_ptr() == output.data_ptr(), in_place)
                self.assertTrue(torch.equal(o, o_ref))
                self.assertTrue(torch.equal(pool, pool_ref))
                if not in_place:
                    self.assertTrue(torch.isnan(output.float()).all())
        # Nothing is written past the rows of the output.
        self.assertTrue(torch.isnan(padded[:, T:].float()).all())

    def test_pipelined_scan_is_bit_identical(self):
        """With SGLANG_GDN_CHUNK_PIPELINE one long sequence runs as up to 4
        parts over two streams; the output, the per-chunk states and the final
        state are the same bits as in one pass, also with a padded tail and a
        part boundary inside the last chunk. Batches of several sequences,
        sequences under 8192 tokens and read-only states keep the one pass."""
        device = get_device()
        dtype = torch.bfloat16
        Hg, H, K, V, pool_size = 2, 6, 128, 128, 4
        torch.manual_seed(0)
        cases = (
            # label, rows, sequence lengths, inplace_update
            ("16384 tokens, 4 parts", 16384, [16384], True),
            ("12600 tokens, 3 parts", 12600, [12600], True),
            ("8192 tokens, 2 parts", 8192, [8192], True),
            ("10000 tokens + padded tail", 10240, [10000], True),
            ("8000 tokens, one pass", 8000, [8000], True),
            ("two sequences, one pass", 18000, [9000, 9000], True),
            ("read-only state, one pass", 16384, [16384], False),
        )
        for label, T, lens, inplace_update in cases:
            with self.subTest(label):
                pool_init = (
                    torch.randn(pool_size, H, V, K, dtype=torch.float32, device=device)
                    * 0.1
                )
                cache_indices = torch.arange(
                    1, len(lens) + 1, dtype=torch.int32, device=device
                )
                cu_seqlens = torch.tensor(
                    [0] + torch.tensor(lens).cumsum(0).tolist(),
                    dtype=torch.long,
                    device=device,
                )
                q = torch.randn(1, T, Hg, K, dtype=dtype, device=device)
                k = torch.randn(1, T, Hg, K, dtype=dtype, device=device)
                v = torch.randn(1, T, H, V, dtype=dtype, device=device)
                g = torch.nn.functional.logsigmoid(
                    torch.randn(1, T, H, dtype=torch.float32, device=device)
                )
                beta = torch.sigmoid(
                    torch.randn(1, T, H, dtype=torch.float32, device=device)
                )
                results = []
                for pipeline in (False, True):
                    pool = pool_init.clone()
                    output = torch.full_like(v, float("nan"))
                    with envs.SGLANG_GDN_CHUNK_PIPELINE.override(pipeline):
                        o, _, h = chunk_gated_delta_rule(
                            q=q,
                            k=k,
                            v=v,
                            g=g,
                            beta=beta,
                            initial_state=pool,
                            initial_state_indices=cache_indices,
                            cu_seqlens=cu_seqlens,
                            head_first=False,
                            use_qk_l2norm_in_kernel=True,
                            inplace_update=inplace_update,
                            output=output,
                        )
                    self.assertEqual(o.data_ptr(), output.data_ptr())
                    results.append((o, h, pool))
                (o0, h0, pool0), (o1, h1, pool1) = results
                self.assertTrue(torch.equal(o1, o0))
                self.assertTrue(torch.equal(h1, h0))
                self.assertTrue(torch.equal(pool1, pool0))
                if sum(lens) < T:
                    self.assertTrue(
                        torch.equal(
                            o1[:, sum(lens) :], torch.zeros_like(o1[:, sum(lens) :])
                        )
                    )

    def test_host_offsets_are_bit_identical_without_syncs(self):
        """With cu_seqlens_cpu the scan plans from the host offsets instead of
        reading cu_seqlens back: the same bits as without it, in one pass and
        pipelined, and no synchronizing call once the kernels are compiled."""
        device = get_device()
        dtype = torch.bfloat16
        Hg, H, K, V, pool_size = 2, 6, 128, 128, 4
        torch.manual_seed(0)
        cases = (
            # label, rows, sequence lengths, pipeline
            ("three sequences, one pass", 3000, [1000, 64, 1936], False),
            ("16384 tokens, pipelined", 16384, [16384], True),
            ("10000 tokens + padded tail, pipelined", 10240, [10000], True),
            ("1696 tokens + padded tail, one pass", 2048, [1696], True),
        )
        for label, T, lens, pipeline in cases:
            with self.subTest(label):
                pool_init = (
                    torch.randn(pool_size, H, V, K, dtype=torch.float32, device=device)
                    * 0.1
                )
                cache_indices = torch.arange(
                    1, len(lens) + 1, dtype=torch.int32, device=device
                )
                q = torch.randn(1, T, Hg, K, dtype=dtype, device=device)
                k = torch.randn(1, T, Hg, K, dtype=dtype, device=device)
                v = torch.randn(1, T, H, V, dtype=dtype, device=device)
                g = torch.nn.functional.logsigmoid(
                    torch.randn(1, T, H, dtype=torch.float32, device=device)
                )
                beta = torch.sigmoid(
                    torch.randn(1, T, H, dtype=torch.float32, device=device)
                )
                offsets = [0, *itertools.accumulate(lens)]

                def run(host):
                    # Fresh offsets per call, as per forward pass: nothing comes
                    # from the index caches of an earlier call.
                    cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device=device)
                    cu_seqlens_cpu = (
                        torch.tensor(offsets, dtype=torch.int32) if host else None
                    )
                    pool = pool_init.clone()
                    output = torch.full_like(v, float("nan"))
                    torch.cuda.synchronize()
                    with envs.SGLANG_GDN_CHUNK_PIPELINE.override(pipeline):
                        if host and torch.cuda.is_available():
                            torch.cuda.set_sync_debug_mode("error")
                        try:
                            o, _, h = chunk_gated_delta_rule(
                                q=q,
                                k=k,
                                v=v,
                                g=g,
                                beta=beta,
                                initial_state=pool,
                                initial_state_indices=cache_indices,
                                cu_seqlens=cu_seqlens,
                                head_first=False,
                                use_qk_l2norm_in_kernel=True,
                                output=output,
                                cu_seqlens_cpu=cu_seqlens_cpu,
                            )
                        finally:
                            if torch.cuda.is_available():
                                torch.cuda.set_sync_debug_mode(0)
                    return o, h, pool

                o0, h0, pool0 = run(host=False)
                o1, h1, pool1 = run(host=True)
                self.assertTrue(torch.equal(o1, o0))
                self.assertTrue(torch.equal(h1, h0))
                self.assertTrue(torch.equal(pool1, pool0))

    def test_chunk_indices_and_offsets_from_host_offsets(self):
        """Chunk indices built from host offsets and chunk offsets padded on
        the device match the device-read versions, values and dtype."""
        from sglang.kernels.ops.attention.fla.index import (
            prepare_chunk_indices,
            prepare_chunk_offsets,
        )

        device = get_device()
        for lens in ([1], [64], [65], [1000, 64, 1936], [5, 0, 3], [16384]):
            for dtype in (torch.int32, torch.long):
                with self.subTest(lens=lens, dtype=dtype):
                    offsets = [0, *itertools.accumulate(lens)]
                    cu_seqlens = torch.tensor(offsets, dtype=dtype, device=device)
                    expected = prepare_chunk_indices(cu_seqlens.clone(), 64)
                    actual = prepare_chunk_indices(
                        cu_seqlens, 64, torch.tensor(offsets, dtype=dtype)
                    )
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assertEqual(actual.device, expected.device)
                    self.assertTrue(torch.equal(actual, expected))

                    lens_t = cu_seqlens[1:] - cu_seqlens[:-1]
                    expected = torch.cat(
                        [cu_seqlens.new_tensor([0]), (lens_t + 63) // 64]
                    ).cumsum(-1)
                    actual = prepare_chunk_offsets(cu_seqlens.clone(), 64)
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assertTrue(torch.equal(actual, expected))

    def test_epilogue_sees_every_row_once_when_final(self):
        """epilogue(o, start, end) covers each output row exactly once, after
        the row is final: per part of a pipelined scan (the side stream's parts
        while chunk_h continues, the last part and any padded tail after it),
        once over all rows in one pass. The output is the same as without it."""
        device = get_device()
        dtype = torch.bfloat16
        Hg, H, K, V, pool_size = 2, 6, 128, 128, 4
        torch.manual_seed(0)
        cases = (
            # label, rows, sequence lengths, expected epilogue calls
            ("16384 tokens, 4 parts", 16384, [16384], 4),
            ("12600 tokens, 3 parts", 12600, [12600], 3),
            ("10000 tokens + padded tail, 2 parts", 10240, [10000], 2),
            ("8000 tokens, one pass", 8000, [8000], 1),
            ("two sequences, one pass", 18000, [9000, 9000], 1),
        )
        for label, T, lens, num_calls in cases:
            with self.subTest(label):
                pool_init = (
                    torch.randn(pool_size, H, V, K, dtype=torch.float32, device=device)
                    * 0.1
                )
                cache_indices = torch.arange(
                    1, len(lens) + 1, dtype=torch.int32, device=device
                )
                cu_seqlens = torch.tensor(
                    [0] + torch.tensor(lens).cumsum(0).tolist(),
                    dtype=torch.long,
                    device=device,
                )
                q = torch.randn(1, T, Hg, K, dtype=dtype, device=device)
                k = torch.randn(1, T, Hg, K, dtype=dtype, device=device)
                v = torch.randn(1, T, H, V, dtype=dtype, device=device)
                g = torch.nn.functional.logsigmoid(
                    torch.randn(1, T, H, dtype=torch.float32, device=device)
                )
                beta = torch.sigmoid(
                    torch.randn(1, T, H, dtype=torch.float32, device=device)
                )
                calls = []
                seen = torch.full_like(v, float("nan"))

                def epilogue(o, start, end):
                    calls.append((start, end))
                    seen[:, start:end].copy_(o[:, start:end])

                results = []
                for hook in (None, epilogue):
                    pool = pool_init.clone()
                    output = torch.full_like(v, float("nan"))
                    with envs.SGLANG_GDN_CHUNK_PIPELINE.override(True):
                        o, _, _ = chunk_gated_delta_rule(
                            q=q,
                            k=k,
                            v=v,
                            g=g,
                            beta=beta,
                            initial_state=pool,
                            initial_state_indices=cache_indices,
                            cu_seqlens=cu_seqlens,
                            head_first=False,
                            use_qk_l2norm_in_kernel=True,
                            output=output,
                            epilogue=hook,
                        )
                    results.append((o, pool))
                (o0, pool0), (o1, pool1) = results
                self.assertTrue(torch.equal(o1, o0))
                self.assertTrue(torch.equal(pool1, pool0))
                if not o1.is_cuda:
                    continue  # the pipelined scan (and its part count) is GPU-only
                self.assertEqual(len(calls), num_calls)
                bounds = sorted(calls)
                self.assertEqual(bounds[0][0], 0)
                self.assertEqual(bounds[-1][1], T)
                for (_, end), (start, _) in zip(bounds, bounds[1:]):
                    self.assertEqual(end, start)
                self.assertTrue(torch.equal(seen, o1))


if __name__ == "__main__":
    unittest.main()
