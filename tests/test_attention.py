"""Check reconstruction against the inference kernel, including GQA and masks."""

import unittest
import torch
from tabknn.attention import _weights


class AttentionTests(unittest.TestCase):
    def test_reconstructed_output_matches_sdpa(self):
        torch.manual_seed(7)
        for kv_heads in [1, 4]:
            q = torch.randn(2, 4, 3, 8)
            k = torch.randn(2, kv_heads, 6, 8)
            v = torch.randn(2, kv_heads, 6, 8)
            mask = torch.ones(3, 6, dtype=torch.bool)
            mask[:, -1] = False
            for attention_mask in [None, mask, torch.randn(3, 6)]:
                weights = _weights(q, k, attention_mask)
                values = v.repeat_interleave(4 // kv_heads, dim=1)
                actual = weights @ values
                expected = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=attention_mask, enable_gqa=kv_heads != 4)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, 3))

    def test_training_permutation_reorders_weights(self):
        torch.manual_seed(8)
        q, k = torch.randn(1, 4, 3, 8), torch.randn(1, 1, 6, 8)
        order = torch.tensor([4, 0, 5, 1, 3, 2])
        torch.testing.assert_close(_weights(q, k[..., order, :]), _weights(q, k)[..., order])


if __name__ == "__main__":
    unittest.main()
