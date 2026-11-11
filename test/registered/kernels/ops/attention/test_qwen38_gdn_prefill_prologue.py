"""Exact parity tests for the Qwen3.8 TP1 GDN fused prefill producer."""

import itertools
import unittest

import torch

from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
from sglang.kernels.ops.attention.linear.fused_gdn_gating import fused_gdn_gating
from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    QWEN38_GDN_HEAD_DIM,
    QWEN38_GDN_NUM_QK_HEADS,
    QWEN38_GDN_NUM_V_HEADS,
    QWEN38_GDN_PROLOGUE_TILE_M,
    QWEN38_GDN_QKV_DIM,
    can_use_qwen38_gdn_prefill_prologue,
    causal_conv1d_fn,
    qwen38_gdn_prefill_prologue,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=90, stage="base-b", runner_config="1-gpu-large")


def _cu_seqlens(lengths: list[int]) -> torch.Tensor:
    return torch.tensor(
        [0, *itertools.accumulate(lengths)],
        dtype=torch.int32,
        device="cuda",
    )


def _make_inputs(
    lengths: list[int],
    *,
    has_initial_state: bool,
    projection_stride: bool = True,
    has_bias: bool = True,
    seed: int = 20260921,
) -> dict[str, torch.Tensor | list[int] | None]:
    torch.manual_seed(seed)
    tokens = sum(lengths)
    if projection_stride:
        qkvz = torch.randn(
            tokens,
            QWEN38_GDN_QKV_DIM + QWEN38_GDN_NUM_V_HEADS * QWEN38_GDN_HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        x = qkvz[:, :QWEN38_GDN_QKV_DIM].transpose(0, 1)
    else:
        x = torch.randn(
            tokens,
            QWEN38_GDN_QKV_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        ).transpose(0, 1)

    ba = torch.randn(
        tokens,
        2 * QWEN38_GDN_NUM_V_HEADS,
        dtype=torch.bfloat16,
        device="cuda",
    )
    raw_b = ba[:, :QWEN38_GDN_NUM_V_HEADS]
    raw_a = ba[:, QWEN38_GDN_NUM_V_HEADS:]
    weight = (
        torch.randn(
            QWEN38_GDN_QKV_DIM,
            4,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.125
    )
    bias = (
        torch.randn(
            QWEN38_GDN_QKV_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.125
        if has_bias
        else None
    )
    num_sequences = len(lengths)
    num_slots = num_sequences + 3
    conv_state = torch.randn(
        num_slots,
        QWEN38_GDN_QKV_DIM,
        3,
        dtype=torch.bfloat16,
        device="cuda",
    )
    cache_indices = torch.arange(
        1,
        num_sequences + 1,
        dtype=torch.int32,
        device="cuda",
    )
    initial = torch.full(
        (num_sequences,),
        has_initial_state,
        dtype=torch.bool,
        device="cuda",
    )
    A_log = torch.randn(
        QWEN38_GDN_NUM_V_HEADS,
        dtype=torch.float32,
        device="cuda",
    )
    dt_bias = torch.randn(
        QWEN38_GDN_NUM_V_HEADS,
        dtype=torch.bfloat16,
        device="cuda",
    )
    return {
        "x": x,
        "weight": weight,
        "bias": bias,
        "conv_state": conv_state,
        "query_start_loc": _cu_seqlens(lengths),
        "seq_lens_cpu": lengths,
        "cache_indices": cache_indices,
        "has_initial_state": initial,
        "A_log": A_log,
        "a": raw_a,
        "b": raw_b,
        "dt_bias": dt_bias,
    }


def _legacy_prologue(
    inputs: dict[str, torch.Tensor | list[int]],
    conv_state: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    mixed_qkv = causal_conv1d_fn(
        inputs["x"],
        inputs["weight"],
        inputs["bias"],
        activation="silu",
        conv_states=conv_state,
        has_initial_state=inputs["has_initial_state"],
        cache_indices=inputs["cache_indices"],
        query_start_loc=inputs["query_start_loc"],
        seq_lens_cpu=inputs["seq_lens_cpu"],
    ).transpose(0, 1)
    tokens = mixed_qkv.shape[0]
    qk_dim = QWEN38_GDN_NUM_QK_HEADS * QWEN38_GDN_HEAD_DIM
    q_raw = (
        mixed_qkv[:, :qk_dim]
        .reshape(1, tokens, QWEN38_GDN_NUM_QK_HEADS, QWEN38_GDN_HEAD_DIM)
        .contiguous()
    )
    k_raw = (
        mixed_qkv[:, qk_dim : 2 * qk_dim]
        .reshape(1, tokens, QWEN38_GDN_NUM_QK_HEADS, QWEN38_GDN_HEAD_DIM)
        .contiguous()
    )
    v = (
        mixed_qkv[:, 2 * qk_dim :]
        .reshape(1, tokens, QWEN38_GDN_NUM_V_HEADS, QWEN38_GDN_HEAD_DIM)
        .contiguous()
    )
    q = l2norm_fwd(q_raw)
    k = l2norm_fwd(k_raw)
    g, beta = fused_gdn_gating(
        inputs["A_log"],
        inputs["a"],
        inputs["b"],
        inputs["dt_bias"],
    )
    return q_raw, k_raw, v, g, beta, q, k


@unittest.skipIf(
    not torch.cuda.is_available() or torch.version.hip is None,
    "Qwen3.8 fused prefill prologue requires ROCm",
)
class TestQwen38GDNPrefillPrologue(unittest.TestCase):
    def _assert_exact(
        self,
        name: str,
        candidate: torch.Tensor,
        reference: torch.Tensor,
    ) -> None:
        self.assertEqual(candidate.shape, reference.shape, name)
        self.assertEqual(candidate.dtype, reference.dtype, name)
        bit_dtype = {
            torch.bfloat16: torch.int16,
            torch.float16: torch.int16,
            torch.float32: torch.int32,
            torch.float64: torch.int64,
        }.get(candidate.dtype)
        if bit_dtype is None:
            candidate_bits = candidate
            reference_bits = reference
        else:
            candidate_bits = candidate.contiguous().view(bit_dtype)
            reference_bits = reference.contiguous().view(bit_dtype)
        if torch.equal(candidate_bits, reference_bits):
            return
        mismatch = candidate_bits != reference_bits
        first_index = tuple(int(index) for index in mismatch.nonzero()[0].tolist())
        diff = (candidate.float() - reference.float()).abs()
        self.fail(
            f"{name} crossed an exact legacy boundary: "
            f"mismatches={int(mismatch.sum().item())}/{mismatch.numel()}, "
            f"first_index={first_index}, "
            f"candidate={candidate[first_index].item()}, "
            f"reference={reference[first_index].item()}, "
            f"candidate_bits={candidate_bits[first_index].item()}, "
            f"reference_bits={reference_bits[first_index].item()}, "
            f"max_abs={diff.max().item()}"
        )

    def _assert_prologue_exact(
        self,
        lengths: list[int],
        *,
        has_initial_state: bool,
        projection_stride: bool = True,
        has_bias: bool = True,
    ) -> None:
        inputs = _make_inputs(
            lengths,
            has_initial_state=has_initial_state,
            projection_stride=projection_stride,
            has_bias=has_bias,
        )
        ref_state = inputs["conv_state"].clone()
        fused_state = inputs["conv_state"].clone()
        expected = _legacy_prologue(inputs, ref_state)
        actual = qwen38_gdn_prefill_prologue(
            inputs["x"],
            inputs["weight"],
            inputs["bias"],
            fused_state,
            inputs["query_start_loc"],
            inputs["seq_lens_cpu"],
            inputs["cache_indices"],
            inputs["has_initial_state"],
            inputs["A_log"],
            inputs["a"],
            inputs["b"],
            inputs["dt_bias"],
        )
        for name, candidate, reference in zip(
            ("q_raw_post_conv", "k_raw_post_conv", "v", "g", "beta"),
            actual,
            expected[:5],
        ):
            self._assert_exact(name, candidate, reference)

        # Localize any future mismatch: raw producer parity is checked above,
        # while this applies the exact legacy L2Norm kernel independently.
        self._assert_exact("q_after_legacy_l2norm", l2norm_fwd(actual[0]), expected[5])
        self._assert_exact("k_after_legacy_l2norm", l2norm_fwd(actual[1]), expected[6])
        self._assert_exact("conv_state", fused_state, ref_state)

    def test_dense_and_packed_varlen_exact(self):
        cases = (
            [64],
            [65],
            [13, 17, 19, 23],
        )
        for lengths in cases:
            for has_initial_state in (False, True):
                with self.subTest(
                    lengths=lengths,
                    has_initial_state=has_initial_state,
                ):
                    self._assert_prologue_exact(
                        lengths,
                        has_initial_state=has_initial_state,
                    )

    def test_without_conv_bias_exact(self):
        """Qwen3.8 checkpoints have no Conv1D bias; the producer must still
        dispatch and match the legacy bias-free Conv1D bit for bit."""
        for lengths in ([13, 17, 19, 23],):
            for has_initial_state in (False, True):
                with self.subTest(
                    lengths=lengths,
                    has_initial_state=has_initial_state,
                ):
                    self._assert_prologue_exact(
                        lengths,
                        has_initial_state=has_initial_state,
                        has_bias=False,
                    )

    def test_conv_bias_contract(self):
        inputs = _make_inputs([8], has_initial_state=False)

        def check(bias: torch.Tensor | None) -> tuple[bool, str]:
            return can_use_qwen38_gdn_prefill_prologue(
                inputs["x"],
                inputs["weight"],
                bias,
                inputs["conv_state"],
                inputs["query_start_loc"],
                inputs["seq_lens_cpu"],
                inputs["cache_indices"],
                inputs["has_initial_state"],
                inputs["A_log"],
                inputs["a"],
                inputs["b"],
                inputs["dt_bias"],
            )

        bias = inputs["bias"]
        self.assertEqual(check(bias), (True, "eligible"))
        self.assertEqual(check(None), (True, "eligible"))
        rejected = {
            "shape": bias[:-1],
            "dtype": bias.float(),
            "stride": bias.repeat_interleave(2)[::2],
        }
        for name, bad_bias in rejected.items():
            with self.subTest(rejected=name):
                self.assertFalse(check(bad_bias)[0])

    def test_tiled_kernel_tile_edges_exact(self):
        """The tiled producer at its token-tile edges: sequences that fill a
        tile, end one past it, are shorter than the conv state or than a tile,
        and a long one that ends inside a tile."""
        tile = QWEN38_GDN_PROLOGUE_TILE_M
        for lengths in (
            [tile],
            [tile + 1],
            [3, 2 * tile + 5, 1],
        ):
            for has_initial_state in (False, True):
                with self.subTest(
                    lengths=lengths,
                    has_initial_state=has_initial_state,
                ):
                    self._assert_prologue_exact(
                        lengths,
                        has_initial_state=has_initial_state,
                    )

    def test_normalize_qk_matches_l2norm(self):
        """With normalize_qk, q and k come out as l2norm_fwd of the raw
        producer output to within one BF16 ulp (the norm is taken in the
        producer's registers instead of a second kernel); v, g, beta and the
        conv state do not change."""
        tile = QWEN38_GDN_PROLOGUE_TILE_M
        for lengths in ([13, 17, 19, 23], [tile + 1, 2]):
            for has_initial_state in (False, True):
                with self.subTest(
                    lengths=lengths,
                    has_initial_state=has_initial_state,
                ):
                    inputs = _make_inputs(lengths, has_initial_state=has_initial_state)
                    raw_state = inputs["conv_state"].clone()
                    normed_state = inputs["conv_state"].clone()

                    def run(state, **kwargs):
                        return qwen38_gdn_prefill_prologue(
                            inputs["x"],
                            inputs["weight"],
                            inputs["bias"],
                            state,
                            inputs["query_start_loc"],
                            inputs["seq_lens_cpu"],
                            inputs["cache_indices"],
                            inputs["has_initial_state"],
                            inputs["A_log"],
                            inputs["a"],
                            inputs["b"],
                            inputs["dt_bias"],
                            **kwargs,
                        )

                    raw = run(raw_state)
                    normed = run(normed_state, normalize_qk=True)
                    for name, candidate, raw_qk in zip("qk", normed, raw):
                        reference = l2norm_fwd(raw_qk)
                        self.assertEqual(candidate.shape, reference.shape, name)
                        ulps = (
                            candidate.view(torch.int16).int()
                            - reference.view(torch.int16).int()
                        ).abs()
                        self.assertLessEqual(int(ulps.max()), 1, name)
                        self.assertLess(float((ulps > 0).float().mean()), 1e-3, name)
                    for name, candidate, reference in zip(
                        ("v", "g", "beta"), normed[2:], raw[2:]
                    ):
                        self._assert_exact(name, candidate, reference)
                    self._assert_exact("conv_state", normed_state, raw_state)


if __name__ == "__main__":
    unittest.main()
