import random

import torch

from lcasr.utils.augmentation import SpecAugment


class RandomMixedMaskingAugment(torch.nn.Module):
    """Random mixed masking policy ported from learning-to-augment RMM."""

    def __init__(
        self,
        zero_masking=True,
        time_masks_min=12,
        time_masks_max=12,
        scale_time_masks_by_seq_len=False,
        time_masks_reference_seq_len=2048,
        time_masks_target_seq_len=None,
        freq_masks_min=5,
        freq_masks_max=7,
        freq_mask_param_min=24,
        freq_mask_param_max=44,
        branch='random',
    ):
        super().__init__()
        valid_branches = {'random', 'time', 'freq', 'time_freq'}
        if branch not in valid_branches:
            raise ValueError(f'rmm_branch must be one of {sorted(valid_branches)}, got {branch!r}')
        self.zero_masking = zero_masking
        self.time_masks_min = time_masks_min
        self.time_masks_max = time_masks_max
        self.scale_time_masks_by_seq_len = scale_time_masks_by_seq_len
        self.time_masks_reference_seq_len = time_masks_reference_seq_len
        self.time_masks_target_seq_len = time_masks_target_seq_len
        self.freq_masks_min = freq_masks_min
        self.freq_masks_max = freq_masks_max
        self.freq_mask_param_min = freq_mask_param_min
        self.freq_mask_param_max = freq_mask_param_max
        self.branch = branch

    def _random_int(self, low, high, name):
        if low > high:
            raise ValueError(f'{name}_min must be <= {name}_max, got {low} > {high}')
        return random.randint(low, high)

    def _scaled_time_mask_count(self, count):
        if not self.scale_time_masks_by_seq_len:
            return count
        if self.time_masks_reference_seq_len <= 0:
            raise ValueError('rmm_time_masks_reference_seq_len must be positive')

        target_seq_len = self.time_masks_target_seq_len
        if target_seq_len is None or target_seq_len <= 0:
            return count
        scale = target_seq_len / self.time_masks_reference_seq_len
        return max(1, int(round(count * scale)))

    def forward(self, spec):
        n_time_masks = self._random_int(self.time_masks_min, self.time_masks_max, 'rmm_time_masks')
        n_time_masks = self._scaled_time_mask_count(n_time_masks)
        min_p = random.random() / 2
        time_masker = SpecAugment(
            n_time_masks=n_time_masks,
            n_freq_masks=0,
            freq_mask_param=0,
            zero_masking=True,
            min_p=min_p,
        )
        n_freq_masks = self._random_int(self.freq_masks_min, self.freq_masks_max, 'rmm_freq_masks')
        freq_mask_param = self._random_int(self.freq_mask_param_min, self.freq_mask_param_max, 'rmm_freq_mask_param')
        freq_masker = SpecAugment(
            n_time_masks=0,
            n_freq_masks=n_freq_masks,
            freq_mask_param=freq_mask_param,
            zero_masking=True,
        )

        mask = torch.ones_like(spec)
        method = self.branch
        if method == 'random':
            method = random.choice(('time', 'freq', 'time_freq'))
        if method == 'time':
            mask = time_masker(mask)
        elif method == 'freq':
            mask = freq_masker(mask)
        else:
            mask = freq_masker(time_masker(mask))

        if self.zero_masking:
            return spec * mask
        return spec * mask + (1 - mask) * spec.mean(dim=(1, 2), keepdim=True)
