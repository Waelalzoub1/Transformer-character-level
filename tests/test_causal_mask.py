"""Causal masking: logits at position t must not depend on tokens after t.

    python -m pytest tests/  or  python tests/test_causal_mask.py
"""
import os, sys, unittest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import CharTransformerLM


class CausalMaskTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = CharTransformerLM(vocab_size=32, d_model=64, n_heads=4, n_layers=2, block_size=16, dropout=0.0).eval()

    def test_future_tokens_do_not_change_earlier_logits(self):
        x = torch.randint(0, 32, (2, 16))
        y = x.clone()
        y[:, 8:] = (y[:, 8:] + 1) % 32          # perturb only positions 8..15
        with torch.no_grad():
            lx, _ = self.model(x)
            ly, _ = self.model(y)
        self.assertTrue(torch.allclose(lx[:, :8], ly[:, :8], atol=1e-6), "logits before the perturbation changed")
        self.assertFalse(torch.allclose(lx[:, 8:], ly[:, 8:]), "perturbed positions should change")

    def test_prefix_logits_match_full_sequence(self):
        x = torch.randint(0, 32, (1, 16))
        with torch.no_grad():
            full, _ = self.model(x)
            prefix, _ = self.model(x[:, :5])
        self.assertTrue(torch.allclose(full[:, :5], prefix, atol=1e-6))

    def test_gradient_does_not_flow_from_future(self):
        x = torch.randint(0, 32, (1, 16))
        emb = self.model.token_emb(x).detach().requires_grad_(True)
        pos = self.model.pos_emb(torch.arange(16))
        h = self.model.blocks(self.model.drop(emb + pos))
        logits = self.model.head(self.model.ln_f(h))
        logits[0, 3].sum().backward()            # loss at position 3 only
        self.assertGreater(emb.grad[0, :4].abs().sum().item(), 0)
        self.assertEqual(emb.grad[0, 4:].abs().sum().item(), 0.0, "gradient leaked from future positions")


if __name__ == "__main__":
    unittest.main()
