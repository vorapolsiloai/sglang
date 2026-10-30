# Adapted from https://github.com/flashinfer-ai/flashinfer/blob/main/tests/test_sampling.py
# and /sgl-workspace/sglang/python/sglang/kernels/aot/tests/test_sampling.py

import sys

import pytest
import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=6, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")

if is_hip():
    from sglang.kernels.ops.sampling.renorm_triton import (
        top_k_renorm_probs_triton as top_k_renorm_prob,
    )
    from sglang.kernels.ops.sampling.renorm_triton import (
        top_k_top_p_renorm_probs_triton,
    )
    from sglang.kernels.ops.sampling.renorm_triton import (
        top_p_renorm_probs_triton as top_p_renorm_prob,
    )
else:
    from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
def test_top_k_renorm_probs(batch_size, vocab_size, k):
    """Test top_k_renorm_probs kernel for correctness.

    This test validates that the kernel correctly:
    1. Identifies the top-k probabilities
    2. Masks out non-top-k values
    3. Renormalizes the remaining probabilities to sum to 1
    """
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, _ = torch.sort(normalized_prob, descending=True)
    pivot = sorted_prob[:, k - 1]
    mask = (normalized_prob >= pivot.unsqueeze(-1)).int()
    renorm_prob_ground_truth = normalized_prob.clone()
    renorm_prob_ground_truth[mask == 0] = 0
    renorm_prob_ground_truth = renorm_prob_ground_truth / renorm_prob_ground_truth.sum(
        dim=-1, keepdim=True
    )

    renorm_prob = top_k_renorm_prob(normalized_prob, k)
    for i in range(batch_size):
        torch.testing.assert_close(
            renorm_prob_ground_truth[i],
            renorm_prob[i],
            rtol=1e-3,
            atol=1e-3,
        )


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_top_p_renorm_probs(batch_size, vocab_size, p):
    """Test top_p_renorm_probs kernel for correctness.

    This test validates that the kernel correctly:
    1. Computes the cumulative probability distribution
    2. Identifies tokens in the top-p threshold
    3. Masks out tokens outside the threshold
    4. Renormalizes the remaining probabilities to sum to 1
    """
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device="cuda:0")
    mask.scatter_add_(1, indices, (cdf >= (1 - p)).int())
    renorm_prob_ground_truth = normalized_prob.clone()
    renorm_prob_ground_truth[mask == 0] = 0
    renorm_prob_ground_truth = renorm_prob_ground_truth / renorm_prob_ground_truth.sum(
        dim=-1, keepdim=True
    )

    renorm_prob = top_p_renorm_prob(normalized_prob, p)
    torch.testing.assert_close(
        renorm_prob_ground_truth,
        renorm_prob,
        rtol=1e-3,
        atol=1e-3,
    )


def _bf16_logit_probs(batch_size, vocab_size, scale, seed):
    """Softmax of bf16-rounded logits: ties at the top-k pivot are common, as in serving."""
    gen = torch.Generator(device="cuda:0").manual_seed(seed)
    logits = torch.randn(batch_size, vocab_size, device="cuda:0", generator=gen)
    return torch.softmax((logits * scale).to(torch.bfloat16).float(), dim=-1)


def _row_params(batch_size, k_choices, p_choices, seed):
    gen = torch.Generator().manual_seed(seed)
    top_ks = torch.tensor(k_choices)[torch.randint(len(k_choices), (batch_size,), generator=gen)]
    top_ps = torch.tensor(p_choices)[torch.randint(len(p_choices), (batch_size,), generator=gen)]
    return top_ks.to("cuda:0"), top_ps.to("cuda:0", torch.float32)


@pytest.mark.skipif(not is_hip(), reason="HIP renorm fallback")
@pytest.mark.parametrize("scale", [1.0, 4.0, 12.0])
@pytest.mark.parametrize("vocab_size", [32000, 248320])
def test_top_k_renorm_window_matches_sort(scale, vocab_size):
    probs = _bf16_logit_probs(64, vocab_size, scale, seed=1)
    top_ks, _ = _row_params(64, [1, 5, 20, 50], [1.0], seed=2)
    max_top_k = int(top_ks.max())
    expected = top_k_renorm_prob(probs, top_ks)
    for window in (max_top_k, 4 * max_top_k):
        assert torch.equal(top_k_renorm_prob(probs, top_ks, window), expected)


@pytest.mark.skipif(not is_hip(), reason="HIP renorm fallback")
@pytest.mark.parametrize("scale", [1.0, 4.0, 12.0])
@pytest.mark.parametrize("vocab_size", [32000, 248320])
def test_top_k_top_p_renorm_matches_two_sorts(scale, vocab_size):
    probs = _bf16_logit_probs(64, vocab_size, scale, seed=3)
    top_ks, top_ps = _row_params(64, [1, 5, 20, 50], [0.5, 0.8, 0.95, 1.0], seed=4)
    expected = top_p_renorm_prob(top_k_renorm_prob(probs, top_ks), top_ps)
    actual = top_k_top_p_renorm_probs_triton(probs, top_ks, top_ps, int(top_ks.max()))
    assert torch.equal(actual > 0, expected > 0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not is_hip(), reason="HIP renorm fallback")
def test_top_k_top_p_renorm_ties_outside_window():
    # 100 tokens share the k-th largest logit, so most of the kept ties fall
    # outside the top-k window and enter the top-p CDF only through their count.
    vocab_size, k = 32000, 20
    logits = torch.full((4, vocab_size), -4.0, device="cuda:0")
    logits[:, : k - 1] = torch.linspace(6.0, 2.0, k - 1, device="cuda:0")
    logits[:, k - 1 : k + 99] = 1.0
    probs = torch.softmax(logits, dim=-1)
    top_ks = torch.full((4,), k, device="cuda:0")
    top_ps = torch.tensor([0.3, 0.9, 0.99, 1.0], device="cuda:0")
    expected = top_p_renorm_prob(top_k_renorm_prob(probs, top_ks), top_ps)
    actual = top_k_top_p_renorm_probs_triton(probs, top_ks, top_ps, k)
    assert torch.equal(actual > 0, expected > 0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
