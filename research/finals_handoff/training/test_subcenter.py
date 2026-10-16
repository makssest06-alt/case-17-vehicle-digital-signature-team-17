import unittest
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from extract_emb import resolve_tf_mode
from head_model import MarginHead, SubcenterMarginHead


class TestSubcenterMarginHead(unittest.TestCase):
    def test_single_subcenter_matches_existing_arcface(self):
        torch.manual_seed(9)
        features = torch.randn(6, 8)
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        baseline = MarginHead(8, 3, "arcface", scale=30.0, margin=0.25)
        candidate = SubcenterMarginHead(8, 3, num_subcenters=1, scale=30.0, margin=0.25)
        with torch.no_grad():
            candidate.weight[:, 0].copy_(baseline.weight)
        torch.testing.assert_close(candidate(features, labels), baseline(features, labels))

    def test_two_centers_take_max_before_target_margin(self):
        module = SubcenterMarginHead(2, 2, num_subcenters=2, scale=1.0, margin=0.0)
        with torch.no_grad():
            module.weight.copy_(torch.tensor([
                [[1.0, 0.0], [0.8, 0.6]],
                [[0.0, 1.0], [0.6, 0.8]],
            ]))
        features = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        labels = torch.tensor([0, 1])
        logits = module.class_logits(features)
        expected_cos = F.linear(
            F.normalize(features, dim=1),
            F.normalize(module.weight, dim=2).reshape(4, 2),
        ).reshape(2, 2, 2).amax(dim=2)
        torch.testing.assert_close(logits, expected_cos)
        torch.testing.assert_close(module(features, labels), F.cross_entropy(expected_cos, labels))

    def test_arcface_margin_is_applied_after_center_reduction_to_target_only(self):
        margin = 0.2
        module = SubcenterMarginHead(3, 3, num_subcenters=2, scale=1.0, margin=margin)
        features = torch.tensor([[0.2, 0.7, 0.1], [0.4, 0.2, 0.9], [0.8, 0.1, 0.2]])
        labels = torch.tensor([0, 1, 2])
        cosine = module.class_logits(features)
        expected_logits = cosine.clone()
        rows = torch.arange(len(labels))
        target_cos = cosine[rows, labels]
        expected_logits[rows, labels] = torch.cos(torch.acos(target_cos) + margin)
        expected = F.cross_entropy(module.s * expected_logits, labels)
        torch.testing.assert_close(module(features, labels), expected)

    def test_subcenter_head_backpropagates_to_features_and_centers(self):
        module = SubcenterMarginHead(4, 3, num_subcenters=2)
        features = torch.randn(6, 4, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        module(features, labels).backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertTrue(torch.isfinite(module.weight.grad).all())
        self.assertGreater(float(module.weight.grad.abs().sum()), 0.0)

    def test_extraction_defaults_to_checkpoint_transform(self):
        self.assertEqual(resolve_tf_mode(None, {"tf_mode": "squash"}), "squash")
        self.assertEqual(resolve_tf_mode(None, {}), "timm")

    def test_conflicting_extraction_transform_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_tf_mode("timm", {"tf_mode": "squash"})


if __name__ == "__main__":
    unittest.main()
