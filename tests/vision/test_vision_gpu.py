"""The image path on a GPU (skipped without one): FlashInfer's ragged non-causal attention against SDPA, the vision
tower in BF16 against its fp32 CPU reference, and the CUDA runner with images (CUDA graphs on) against the same runner
on CPU."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu(reason="needs a CUDA GPU and the kernel library")


def test_flashinfer_vision_attention_matches_sdpa() -> None:
    from llm_mini_infer.kernels.flashinfer import FlashInferVisionAttention
    from llm_mini_infer.models.qwen3_5.vision import EncodeMeta, TorchVisionAttention

    torch.manual_seed(0)
    cu = [0, 336, 336 + 1056, 336 + 1056 + 64]
    n, h, d = cu[-1], 16, 72
    q, k, v = (torch.randn(n, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    meta = EncodeMeta(n=n, cu=torch.tensor(cu, dtype=torch.int32, device="cuda"), cu_host=cu, cos=None, sin=None,
                      pos_idx=None, pos_w=None)
    fi = FlashInferVisionAttention(h, d, torch.bfloat16, torch.device("cuda"))
    fi.plan(meta)
    assert fi.pad == 128  # FlashInfer has no head dim 72: zero-padded, with the true dim's softmax scale
    padded = [torch.nn.functional.pad(x, (0, fi.pad - d)) for x in (q, k, v)]
    out = fi.run(meta, *padded)[..., :d].float()
    TorchVisionAttention().plan(meta)
    want = TorchVisionAttention().run(meta, q.float(), k.float(), v.float())
    err = (out - want).norm() / want.norm()
    assert err < 1e-2, err.item()


def test_tower_bf16_on_cuda_matches_the_fp32_reference(tmp_path) -> None:
    from llm_mini_infer.kernels.flashinfer import FlashInferVisionAttention
    from llm_mini_infer.models.qwen3_5.vision import TorchVisionAttention, load_vision

    from ..support.vision_tower import images, make_tower

    make_tower(tmp_path)
    ref = load_vision(tmp_path, torch.device("cpu"), torch.float32)
    gpu = load_vision(tmp_path, torch.device("cuda"), torch.bfloat16)
    pxs = images(3)
    grids = [(1, px.shape[1] // 16, px.shape[2] // 16) for px in pxs]
    want = ref(torch.cat([ref.patches(px) for px in pxs]), ref.meta(grids, torch.device("cpu")),
               TorchVisionAttention())
    p = torch.cat([gpu.patches(px.cuda()) for px in pxs])
    assert torch.equal(p.cpu(), torch.cat([ref.patches(px) for px in pxs]))  # the normalisation is exact on CUDA too
    meta = gpu.meta(grids, torch.device("cuda"))
    attn = FlashInferVisionAttention(4, 16, torch.bfloat16, torch.device("cuda"))
    attn.plan(meta)
    got = gpu(p, meta, attn).float().cpu()
    err = (got - want).norm() / want.norm()
    assert err < 3e-2, err.item()


@pytest.mark.parametrize("tower", ["qwen", "gemma"])
def test_the_tower_copies_its_patch_offsets_from_pinned_memory(tmp_path, monkeypatch, tower) -> None:
    """An encoder call's metadata reaches the device with no copy from pageable memory (the driver stages one itself
    and may wait for the stream to do it: a host stall while the step before runs): every host-to-device `Tensor.to`
    in `meta`, the patch offsets', has a page-locked source, on both towers."""
    dev = torch.device("cuda")
    if tower == "qwen":
        from llm_mini_infer.models.qwen3_5.vision import load_vision

        from ..support.vision_tower import make_tower

        make_tower(tmp_path)
        enc, grids = load_vision(tmp_path, dev, torch.bfloat16), [(1, 4, 6), (1, 8, 2)]
    else:
        from llm_mini_infer.models.gemma4.vision import load_gemma_vision

        from ..support.tiny_gemma import make_tiny_gemma_mm

        enc = load_gemma_vision(make_tiny_gemma_mm(tmp_path, depth=1), dev, torch.bfloat16)
        grids = [(1, 6, 9), (1, 3, 3)]
    enc.meta(grids, dev)  # the grids' per-patch tensors cached on the device, as in serving: only the offsets move
    to, pinned = torch.Tensor.to, []

    def spy(self, *args, **kwargs):
        out = to(self, *args, **kwargs)
        if self.device.type == "cpu" and out.device.type == "cuda":
            pinned.append(self.is_pinned())
        return out

    monkeypatch.setattr("torch.Tensor.to", spy)
    meta = enc.meta(grids, dev)
    monkeypatch.undo()
    assert pinned and all(pinned), pinned
    assert meta.cu.device.type == "cuda" and meta.cu.dtype == torch.int32 and meta.cu.cpu().tolist() == meta.cu_host


@pytest.mark.parametrize("k", [0, 2])
def test_cuda_runner_with_images_matches_cpu(tmp_path, k) -> None:
    """The tiny multimodal model through the engine loop on CUDA (BF16, CUDA graphs, FlashInfer) and on CPU (fp32):
    prompt logits close (BF16 noise), with an image split across 8-token chunks (breakable graphs) and MTP."""
    from ..support.qwen_images import Engine, Img, prompt_of
    from ..support.tiny_qwen import make_tiny_vl_model

    path = tmp_path / "vl"
    make_tiny_vl_model(path, seed=4, mtp=True, gpu=True)
    ids, spans = prompt_of([[3, 5, 7, 9, 11], Img(4, 6, 2), [13, 15, 17, 19]])
    outs = {}
    for dev in ("cpu", "cuda"):
        e = Engine(path, tmp_path, budget=8, k=k, device=dev)
        e.add(0, ids, spans, max_tokens=6)
        e.run()
        outs[dev] = (e.prompt_logits(0, len(ids)).cpu(), e.out[0])
    a, b = outs["cpu"][0], outs["cuda"][0]
    assert ((a - b).abs().max() / a.abs().max()) < 5e-2


def test_vision_graphs_match_eager(tmp_path) -> None:
    """The tower replayed from breakable graphs (two buckets; batches that pad) against the eager tower."""
    from llm_mini_infer.kernels.flashinfer import FlashInferVisionAttention
    from llm_mini_infer.models.qwen3_5.vision import load_vision
    from llm_mini_infer.runner.cuda.graphs import no_gc
    from llm_mini_infer.runner.cuda.images import VisionGraphs

    from ..support.vision_tower import images, make_tower

    make_tower(tmp_path)
    enc = load_vision(tmp_path, torch.device("cuda"), torch.bfloat16)
    attn = FlashInferVisionAttention(4, 16, torch.bfloat16, torch.device("cuda"))
    graphs = VisionGraphs(enc, attn, [64, 256], torch.cuda.graph_pool_handle())
    s = torch.cuda.Stream()
    with no_gc(), torch.cuda.stream(s):
        for b in (256, 64):
            graphs.capture(b)
    torch.cuda.synchronize()
    for pxs in (images(5)[:1], images(6)):  # 24 patches; 24 + 16 patches
        grids = [(1, px.shape[1] // 16, px.shape[2] // 16) for px in pxs]
        patches = torch.cat([enc.patches(px.cuda()) for px in pxs])
        meta = enc.meta(grids, torch.device("cuda"))
        got = graphs.run(patches, meta).float().clone()
        attn.plan(meta)
        want = enc(patches, meta, attn).float()
        assert ((got - want).norm() / want.norm()) < 1e-2
