"""gfx950 tests for the MXFP4 quant producers that feed the AITER ASM FP4 GEMM
an AsmMxfp4Input: the marker path against the scheme's own quantization, the
fused Gemma add+RMSNorm+MXFP4 norm and the fused SiLU+mul+MXFP4 MLP producer
against the unfused bf16 path they replace (GemmaRMSNorm / SiluAndMul, then
per_1x32_f4_quant): bit-identical (the SiLU producer with an AITER that has
round_to_input_dtype)."""

import inspect
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from sglang.srt.utils import get_bool_env_var, is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

_RUNNABLE = is_hip() and is_gfx95_supported() and get_bool_env_var("SGLANG_USE_AITER")
# Qwen3.8-27B: hidden 5120, TP1 intermediate 17408.
_HIDDEN, _INTERMEDIATE = 5120, 17408
_BATCHES = (1, 7, 64, 300)


def _asm_layer(scheme, n, k, gen):
    layer = torch.nn.Module()
    weight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=gen)
    # E8M0 2^-7..2^0 keeps the outputs well inside bf16 range.
    scale = torch.randint(120, 128, (n, k // 32), dtype=torch.uint8, generator=gen)
    layer.weight = torch.nn.Parameter(weight.cuda(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scale.cuda(), requires_grad=False)
    scheme.process_weights_after_loading(layer)
    assert layer.use_aiter_asm_fp4_gemm
    return layer


def _row_major_scale(scale, m):
    rows, cols = scale.shape
    return (
        scale.view(torch.uint8)
        .view(rows // 32, cols // 8, 4, 16, 2, 2, 1)
        .permute(0, 5, 3, 1, 4, 2, 6)
        .reshape(rows, cols)[:m]
    )


def _edge_rows(x):
    """Rows of ``x`` with all-zero, tiny (< 1e-10) and large 32-wide blocks."""
    x = x.clone()
    x[0, :64] = 0
    x[-1, 32:64] = 1e-12
    x[-1, 64:96] *= 1e4
    return x


def _rel_fro(out, ref):
    """Relative Frobenius distance between two GEMM outputs."""
    return ((out - ref).norm() / ref.norm()).item()


def _dequant(x_q, scale, values):
    """fp4x2 codes and swizzled E8M0 scales to float32, plus per-element scales."""
    m, k_packed = x_q.shape
    codes = x_q.view(torch.uint8)
    lut = torch.tensor(values, dtype=torch.float32, device=codes.device)
    vals = torch.empty(m, k_packed * 2, dtype=torch.float32, device=codes.device)
    vals[:, 0::2] = lut[(codes & 0xF).long()]
    vals[:, 1::2] = lut[(codes >> 4).long()]
    block = torch.exp2(_row_major_scale(scale, m).float() - 127.0)
    block = block.repeat_interleave(32, dim=1)
    return vals * block, block


@unittest.skipUnless(_RUNNABLE, "requires HIP gfx950 with SGLANG_USE_AITER=1")
class TestMxfp4AsmProducersGpu(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.quantization.quark.schemes import quark_w4a4_mxfp4

        cls.q = quark_w4a4_mxfp4
        cls.scheme = quark_w4a4_mxfp4.QuarkW4A4MXFP4({}, {})
        cls.scheme.out_dtype = torch.bfloat16
        gen = torch.Generator().manual_seed(0)
        with mock.patch.object(quark_w4a4_mxfp4, "_use_aiter_asm_fp4_gemm", True):
            cls.up = _asm_layer(cls.scheme, 1024, _HIDDEN, gen)
            cls.down = _asm_layer(cls.scheme, _HIDDEN, _INTERMEDIATE, gen)

    def _assert_bit_identical(self, fused, ref_input, name):
        """``fused`` equals per_1x32_f4_quant(ref_input) byte for byte, and so
        do the ASM GEMM outputs."""
        m = ref_input.shape[0]
        ref = self.q.AsmMxfp4Input(*self.q.per_1x32_f4_quant(ref_input))
        self.assertTrue(
            torch.equal(fused.x_fp4.view(torch.uint8), ref.x_fp4.view(torch.uint8)),
            f"{name}: fp4 bytes differ",
        )
        self.assertEqual(fused.x_scales.shape, ref.x_scales.shape, name)
        self.assertTrue(
            torch.equal(
                _row_major_scale(fused.x_scales, m),
                _row_major_scale(ref.x_scales.view(torch.uint8), m),
            ),
            f"{name}: scales differ",
        )
        return ref

    def _compare_quant(self, fused, ref_input, y_true, name, roundup=True):
        """``fused`` vs per_1x32_f4_quant(ref_input): with RoundUp scales, within
        one FP4 step of the block scale elementwise and against the fp32 value.
        Returns the exact-match fractions (1.0 = bit-exact)."""
        m = ref_input.shape[0]
        ref = self.q.per_1x32_f4_quant(ref_input)
        deq_f, scale_f = _dequant(fused.x_fp4, fused.x_scales, self.q._MXFP4_VALUES)
        deq_r, scale_r = _dequant(ref[0], ref[1], self.q._MXFP4_VALUES)
        # The largest e2m1 step (4 -> 6) is 2x the block scale; half of it is
        # the worst rounding error of either quantization against y_true.
        if roundup:
            step = torch.maximum(scale_f, scale_r)
            self.assertTrue(bool(((deq_f - deq_r).abs() <= 2 * step).all()), name)
            slack = 1e-2 * y_true.abs() + 1e-6
            err = (deq_f - y_true).abs()
            self.assertTrue(bool((err <= scale_f + slack).all()), name)
        code_match = (
            (fused.x_fp4.view(torch.uint8) == ref[0].view(torch.uint8)).float().mean()
        )
        scale_match = (
            (
                _row_major_scale(fused.x_scales, m)
                == _row_major_scale(ref[1].view(torch.uint8), m)
            )
            .float()
            .mean()
        )
        print(f"{name}: fp4 bytes {code_match:.5f}, scales {scale_match:.5f} equal")
        return code_match.item(), scale_match.item()

    def test_marker_matches_tensor_input(self):
        # Same quantization through the marker (no re-swizzle): bit-exact.
        for m in _BATCHES:
            with self.subTest(m=m):
                x = torch.randn(m, _HIDDEN, dtype=torch.bfloat16, device="cuda")
                marker = self.q.AsmMxfp4Input(*self.q.per_1x32_f4_quant(x))
                torch.testing.assert_close(
                    self.scheme.apply_weights(self.up, marker),
                    self.scheme.apply_weights(self.up, x),
                    rtol=0,
                    atol=0,
                )

    def test_norm_producer_matches_unfused(self):
        from sglang.srt.layers.layernorm import GemmaRMSNorm
        from sglang.srt.models.qwen3_5 import _GemmaRMSNormMXFP4Quant

        fused_norm = _GemmaRMSNormMXFP4Quant(_HIDDEN, 1e-6, lambda num_tokens: True)
        ref_norm = GemmaRMSNorm(_HIDDEN, eps=1e-6)
        weight = 0.1 * torch.randn(_HIDDEN, dtype=torch.bfloat16)
        weight[:64] = -1.0  # (1 + weight) == 0: all-zero output blocks
        for norm in (fused_norm, ref_norm):
            norm.to("cuda", torch.bfloat16)
            norm.weight.data.copy_(weight)
        for m in _BATCHES:
            for edge in (False, True):
                with self.subTest(m=m, edge=edge):
                    x = torch.randn(m, _HIDDEN, dtype=torch.bfloat16, device="cuda")
                    residual = torch.randn_like(x)
                    if edge:
                        x, residual = _edge_rows(x), _edge_rows(residual)
                    out, residual_out = fused_norm.forward_hip(
                        x.clone(), residual.clone()
                    )
                    self.assertIsInstance(out, self.q.AsmMxfp4Input)
                    y, residual_ref = ref_norm.forward_hip(x.clone(), residual.clone())
                    torch.testing.assert_close(
                        residual_out, residual_ref, rtol=0, atol=0
                    )
                    # The producer is the bf16 path's norm kernel with a quant
                    # epilogue: bit-identical, down to the GEMM output.
                    ref = self._assert_bit_identical(out, y, f"norm m={m}")
                    torch.testing.assert_close(
                        self.scheme.apply_weights(self.up, out),
                        self.scheme.apply_weights(self.up, ref),
                        rtol=0,
                        atol=0,
                    )

    def test_silu_producer_matches_unfused(self):
        from aiter.ops.triton import activation

        from sglang.srt.layers.activation import SiluAndMul
        from sglang.srt.models.qwen2_moe import Qwen2MoeMLP

        # AITER honours scaling_mode="ceil" (RoundUp, as per_1x32_f4_quant) and
        # round_to_input_dtype only after the follow-up PR; before it the
        # producer quantizes the fp32 product, with "even" scales if "ceil" is
        # not known.
        roundup = "ceil" in getattr(activation, "_MXFP4_SCALING_MODES", {})
        exact = roundup and "round_to_input_dtype" in (
            inspect.signature(activation.act_mul_and_mxfp4_quant).parameters
        )
        silu_and_mul = SiluAndMul()
        for m in _BATCHES:
            for edge in (False, True):
                with self.subTest(m=m, edge=edge):
                    gate_up = torch.randn(
                        m, 2 * _INTERMEDIATE, dtype=torch.bfloat16, device="cuda"
                    )
                    if edge:
                        gate_up = _edge_rows(gate_up)
                    out = Qwen2MoeMLP._silu_mxfp4_quant_fused(None, gate_up)
                    # What the unfused path quantizes: SiluAndMul's bf16 output.
                    y = silu_and_mul(gate_up)
                    name = f"silu m={m} roundup={roundup} exact={exact}"
                    if exact:
                        ref = self._assert_bit_identical(out, y, name)
                        torch.testing.assert_close(
                            self.scheme.apply_weights(self.down, out),
                            self.scheme.apply_weights(self.down, ref),
                            rtol=0,
                            atol=0,
                        )
                        continue
                    gate, up = gate_up.float().chunk(2, dim=-1)
                    y_true = F.silu(gate) * up
                    code_match, scale_match = self._compare_quant(
                        out, y, y_true, name, roundup
                    )
                    if roundup:
                        self.assertGreaterEqual(scale_match, 0.99)
                        self.assertGreaterEqual(code_match, 0.98)
                    ref_out = self.scheme.apply_weights(self.down, y).float()
                    out_gemm = self.scheme.apply_weights(self.down, out).float()
                    # "even" scales differ from RoundUp on ~22% of the blocks.
                    bound = 0.06 if roundup else 0.15
                    self.assertLess(_rel_fro(out_gemm, ref_out), bound, name)

if __name__ == "__main__":
    unittest.main()
