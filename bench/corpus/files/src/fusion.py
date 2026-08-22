"""Stereo fusion module.

When fusion is on, per-view DINOv2 (or backbone) features are concatenated
and passed through a 1x1 projection before the depth head. When fusion is off,
only the reference view is used.

This is the module referenced by configs as `fusion: true|false`.
"""


def project(features, fusion_on: bool):
    if not fusion_on:
        return features[0]
    import torch
    return torch.cat(features, dim=1)
