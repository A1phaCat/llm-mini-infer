"""The vision tower against transformers' `Qwen3_5VisionModel` (the reference) on a tiny random tower with the real
patch geometry (patch 16, temporal 2, merge 2), on CPU: pixels to patches exactly, embeddings within rounding."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from llm_mini_infer.models.qwen3_5.vision import TorchVisionAttention, load_vision

from ..support.vision_tower import hf_patches, images, make_tower


@pytest.fixture(scope="module")
def tower(tmp_path_factory):
    path = tmp_path_factory.mktemp("tower")
    return path, make_tower(path)


def test_patches_are_hfs_pixel_values(tower) -> None:
    path, _ = tower
    enc = load_vision(path, torch.device("cpu"), torch.float32)
    for px in images():
        assert torch.equal(enc.patches(px), hf_patches(px))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_embeddings_match_transformers(tower, dtype) -> None:
    path, model = tower
    enc = load_vision(path, torch.device("cpu"), dtype)
    hf = model.to(dtype)
    pxs = images(1)
    grids = [(1, px.shape[1] // 16, px.shape[2] // 16) for px in pxs]
    patches = torch.cat([enc.patches(px) for px in pxs])
    meta = enc.meta(grids, torch.device("cpu"))
    ours = enc(patches, meta, TorchVisionAttention()).float()
    with torch.no_grad():
        ref = hf(patches.to(dtype), grid_thw=torch.tensor(grids)).pooler_output.float()
    assert ours.shape == ref.shape == (sum(h * w // 4 for _, h, w in grids), 48)
    err = (ours - ref).norm() / ref.norm()
    assert err < (1e-6 if dtype == torch.float32 else 2e-2), err.item()


def test_one_batch_equals_images_alone(tower) -> None:
    """Attention stays within each image: encoding two images together equals encoding each alone (fp32)."""
    path, _ = tower
    enc = load_vision(path, torch.device("cpu"), torch.float32)
    pxs = images(2)
    grids = [(1, px.shape[1] // 16, px.shape[2] // 16) for px in pxs]
    att = TorchVisionAttention()
    both = enc(torch.cat([enc.patches(px) for px in pxs]), enc.meta(grids, torch.device("cpu")), att)
    alone = torch.cat([enc(enc.patches(px), enc.meta([g], torch.device("cpu")), att) for px, g in zip(pxs, grids)])
    np.testing.assert_allclose(both.numpy(), alone.numpy(), rtol=0, atol=1e-5)


def test_the_patch_offsets_are_the_grids_patch_counts(tower, monkeypatch) -> None:
    """The encoder call's patch offsets: each image's patches after those of the images before it, int32 on the
    tower's device, and on a CPU never pinned (that needs CUDA; tests/vision/test_vision_gpu.py checks the copy on
    one)."""
    path, _ = tower
    enc = load_vision(path, torch.device("cpu"), torch.float32)

    def pin_memory(self, *args, **kwargs):
        raise AssertionError("pinned on a CPU")

    monkeypatch.setattr(torch.Tensor, "pin_memory", pin_memory)
    meta = enc.meta([(1, 4, 6), (1, 2, 2), (1, 8, 2)], torch.device("cpu"))
    assert meta.cu_host == [0, 24, 28, 44] and meta.n == 44
    assert meta.cu.dtype == torch.int32 and meta.cu.device.type == "cpu" and meta.cu.tolist() == meta.cu_host
