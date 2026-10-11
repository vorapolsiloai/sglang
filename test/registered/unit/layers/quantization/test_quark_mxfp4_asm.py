"""CPU-only tests for Quark MXFP4 AITER ASM layout helpers."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization.quark.schemes import quark_w4a4_mxfp4
from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (
    AsmMxfp4Input,
    _asm_fp4_prequantized_input,
    _asm_fp4_scale_swizzle_supported,
    _shuffle_asm_fp4_act_scale,
    _swizzle_asm_fp4_scale,
)
from sglang.test.test_utils import CustomTestCase


class TestQuarkMxfp4AsmScaleLayout(CustomTestCase):
    def test_supported_shape(self):
        scale = torch.empty((32, 8), dtype=torch.uint8)
        self.assertTrue(_asm_fp4_scale_swizzle_supported(scale))

    def test_rejects_unsupported_shapes(self):
        self.assertFalse(
            _asm_fp4_scale_swizzle_supported(torch.empty((16, 8), dtype=torch.uint8))
        )
        self.assertFalse(
            _asm_fp4_scale_swizzle_supported(torch.empty((32, 4), dtype=torch.uint8))
        )
        self.assertFalse(
            _asm_fp4_scale_swizzle_supported(torch.empty((1, 32, 8), dtype=torch.uint8))
        )

    def test_swizzle_matches_aiter_tile_permutation(self):
        scale = torch.arange(32 * 8, dtype=torch.int32).view(32, 8)
        expected = (
            scale.view(1, 2, 16, 1, 2, 4, 1)
            .permute(0, 3, 5, 2, 4, 1, 6)
            .contiguous()
            .view(32, 8)
        )
        actual = _swizzle_asm_fp4_scale(scale)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.shape, scale.shape)
        self.assertTrue(actual.is_contiguous())


class TestQuarkMxfp4AsmTupleInput(CustomTestCase):
    def test_act_scale_shuffle_matches_aiter_padded_layout(self):
        """Row-major (M, K/32) activation scales from the Triton fused-quant
        kernels must land in the (pad256(M), pad8(K/32)) tile order that
        per_1x32_f4_quant(shuffle=True) emits, with E8M0 1.0 in the padding."""
        rows, cols = 3, 6
        scale = torch.randint(0, 0x7F, (rows, cols), dtype=torch.uint8)

        shuffled = _shuffle_asm_fp4_act_scale(scale)

        self.assertEqual(shuffled.dtype, torch.float8_e8m0fnu)
        self.assertEqual(tuple(shuffled.shape), (256, 8))
        # Invert the AITER tile permutation (0, 3, 5, 2, 4, 1, 6).
        unshuffled = (
            shuffled.view(torch.uint8)
            .view(256 // 32, 8 // 8, 4, 16, 2, 2, 1)
            .permute(0, 5, 3, 1, 4, 2, 6)
            .reshape(256, 8)
        )
        expected = torch.full((256, 8), 0x7F, dtype=torch.uint8)
        expected[:rows, :cols] = scale
        torch.testing.assert_close(unshuffled, expected, rtol=0, atol=0)

    def test_only_quantized_pair_is_accepted(self):
        rows, k = 4, 64
        x_q = torch.zeros((rows, k // 2), dtype=torch.uint8)
        x_scales = torch.full((rows, k // 32), 0x7F, dtype=torch.uint8)

        a, a_scales = _asm_fp4_prequantized_input((x_q, x_scales))
        self.assertEqual(a.dtype, torch.float4_e2m1fn_x2)
        self.assertEqual(a.data_ptr(), x_q.data_ptr())
        self.assertEqual(tuple(a_scales.shape), (256, 8))

        # The 3- and 5-tuples ask for Triton-only fused epilogues; they must not
        # be silently unpacked as (x, x_scales).
        for fused in ((x_q, x_scales, torch.empty(0)), (x_q,) * 5):
            with self.assertRaises(NotImplementedError):
                _asm_fp4_prequantized_input(fused)


def _aiter_triton_shuffle_offsets(rows, cols):
    """Flat scale offsets written by AITER's Triton MXFP4 quant kernels with
    shuffle=True (_act_mul_and_dynamic_mxfp4_quant_kernel, SHUFFLE branch)."""
    r = torch.arange(rows).view(-1, 1)
    c = torch.arange(cols).view(1, -1)
    r0, r1, r2 = r // 32, (r % 32) // 16, (r % 32) % 16
    c3, c4, c5 = c // 8, (c % 8) // 4, (c % 8) % 4
    return r1 + c4 * 2 + r2 * 4 + c5 * 64 + c3 * 256 + r0 * 32 * cols


class TestQuarkMxfp4AsmMarkerInput(CustomTestCase):
    """AsmMxfp4Input carries scales already in gemm_a4w4's layout; the scheme
    must use them as-is instead of swizzling them a second time."""

    M, K = 3, 512

    def _marker(self, m=M, k=K, rows=None, cols=None, x_dtype=torch.uint8):
        rows = (m + 255) // 256 * 256 if rows is None else rows
        cols = k // 32 if cols is None else cols
        x_q = torch.zeros((m, k // 2), dtype=x_dtype)
        x_scales = torch.randint(0, 0x7F, (rows, cols), dtype=torch.uint8)
        return AsmMxfp4Input(x_q, x_scales)

    def test_producer_shuffle_layout_matches_scheme_swizzle(self):
        # Layout contract: the AITER producers' shuffled scales are exactly
        # _swizzle_asm_fp4_scale of the row-major scales.
        rows, cols = 256, 16
        scale = torch.randint(0, 256, (rows, cols), dtype=torch.uint8)
        produced = torch.empty(rows * cols, dtype=torch.uint8)
        produced[_aiter_triton_shuffle_offsets(rows, cols).flatten()] = scale.flatten()
        torch.testing.assert_close(
            produced.view(rows, cols), _swizzle_asm_fp4_scale(scale), rtol=0, atol=0
        )

    def test_marker_is_passed_through_without_reswizzle(self):
        marker = self._marker()
        a, a_scales = _asm_fp4_prequantized_input(marker, in_features=self.K)
        self.assertEqual(a.dtype, torch.float4_e2m1fn_x2)
        self.assertEqual(a_scales.dtype, torch.float8_e8m0fnu)
        self.assertEqual(a.data_ptr(), marker.x_fp4.data_ptr())
        self.assertEqual(a_scales.data_ptr(), marker.x_scales.data_ptr())
        torch.testing.assert_close(
            a_scales.view(torch.uint8), marker.x_scales, rtol=0, atol=0
        )

    def test_plain_tuple_is_still_reswizzled(self):
        x_q = torch.zeros((self.M, self.K // 2), dtype=torch.uint8)
        x_scales = torch.randint(0, 0x7F, (self.M, self.K // 32), dtype=torch.uint8)
        _, a_scales = _asm_fp4_prequantized_input((x_q, x_scales))
        torch.testing.assert_close(
            a_scales, _shuffle_asm_fp4_act_scale(x_scales), rtol=0, atol=0
        )

    def test_marker_validation(self):
        m, k = self.M, self.K
        x_q = torch.zeros((m, k // 2), dtype=torch.uint8)
        good = torch.zeros((256, k // 32), dtype=torch.uint8)
        bad = {
            "row-major scales": self._marker(rows=m),
            "unpadded scale cols": self._marker(cols=k // 32 - 8),
            "bf16 activation": self._marker(x_dtype=torch.bfloat16),
            "3D activation": AsmMxfp4Input(x_q.view(1, m, -1), good),
            "non-contiguous scales": AsmMxfp4Input(
                x_q, torch.zeros((k // 32, 256), dtype=torch.uint8).t()
            ),
            "K not a multiple of 256": self._marker(k=128),
        }
        for name, marker in bad.items():
            with self.subTest(name), self.assertRaises(ValueError):
                _asm_fp4_prequantized_input(marker)
        with self.assertRaises(ValueError):
            _asm_fp4_prequantized_input(self._marker(), in_features=2 * k)
        # M on a 256-row boundary needs no extra tile.
        _asm_fp4_prequantized_input(self._marker(m=256), in_features=k)

    def test_non_asm_layer_rejects_marker(self):
        scheme = quark_w4a4_mxfp4.QuarkW4A4MXFP4({}, {})
        for attrs in ({"use_aiter_asm_fp4_gemm": False}, {"dequantized_bf16": True}):
            layer = torch.nn.Module()
            for name, value in attrs.items():
                setattr(layer, name, value)
            with self.subTest(attrs), self.assertRaises(TypeError):
                scheme.apply_weights(layer, self._marker())


class TestQuarkMxfp4AsmNarrowLayer(CustomTestCase):
    def test_narrow_layer_stays_on_triton(self):
        """With the ASM flag on, a layer with N <= 128 (Qwen3.8 GDN in_proj_ba,
        N=96) must keep the Triton GEMM: AITER's untuned ASM heuristic returns
        wrong rows 1..M-1 for it at small batch sizes."""
        n, k = 96, 5120
        layer = torch.nn.Module()
        weight = torch.nn.Parameter(
            torch.zeros((n, k // 2), dtype=torch.uint8), requires_grad=False
        )
        layer.weight = weight
        layer.weight_scale = torch.nn.Parameter(
            torch.zeros((n, k // 32), dtype=torch.uint8), requires_grad=False
        )
        scheme = quark_w4a4_mxfp4.QuarkW4A4MXFP4({}, {})

        with (
            mock.patch.object(quark_w4a4_mxfp4, "_use_aiter_asm_fp4_gemm", True),
            mock.patch.object(
                quark_w4a4_mxfp4, "is_gfx95_supported", return_value=True
            ),
        ):
            scheme.process_weights_after_loading(layer)

        self.assertFalse(layer.use_aiter_asm_fp4_gemm)
        self.assertIs(layer.weight, weight)


if __name__ == "__main__":
    unittest.main()
