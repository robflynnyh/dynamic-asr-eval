import argparse
import sys
import unittest
from pathlib import Path

import torch

_LCASR_ROOT = Path(__file__).resolve().parents[1]
if str(_LCASR_ROOT) not in sys.path:
    sys.path.insert(0, str(_LCASR_ROOT))

import lib


class TestRMMAugmentation(unittest.TestCase):
    def test_rmm_policy_changes_shape_preserving_mask(self):
        args = argparse.Namespace(
            augmentation_policy="rmm",
            rmm_time_masks_min=1,
            rmm_time_masks_max=1,
            rmm_freq_masks_min=1,
            rmm_freq_masks_max=1,
            rmm_freq_mask_param_min=2,
            rmm_freq_mask_param_max=2,
            rmm_zero_masking=True,
        )
        torch.manual_seed(0)
        augment = lib.build_self_training_augmentation(args)
        spec = torch.ones(2, 80, 64)
        augmented = augment(spec)

        self.assertEqual(augmented.shape, spec.shape)
        self.assertTrue(torch.all((augmented == 0) | (augmented == 1)))
        self.assertLessEqual(augmented.sum().item(), spec.sum().item())


if __name__ == "__main__":
    unittest.main()
