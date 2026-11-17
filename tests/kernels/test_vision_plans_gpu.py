"""The vision tower's attention plans in flight on a GPU (skipped without CUDA, FlashInfer and the kernel library).
The host makes encoder calls while the stream sleeps (`torch.cuda._sleep`), so each plan is made before the device has
run the copies of the plans before it: the runner does the same when it queues a step's ENCODE while the step before
still runs, with one tower call per image under batch invariance (`ImageEncoder.per_image`). Each image's output must
equal, bit for bit, its own call's with nothing in flight: FlashInfer's ragged attention for three images in a step
(2,520 patches, then 594 twice: their plans differ) and for two images in each of two steps; then `ImageEncoder.encode`
of three images on the tiny multimodal model, eagerly and from the tower's graphs. tests/kernels/test_vision_plans.py
checks the same against a model of the stream on CPU."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.gpu, pytest.mark.flashinfer(reason="needs a CUDA GPU, FlashInfer and the kernel library")]
SLEEP = 10 ** 8  # GPU cycles (tens of milliseconds) the stream sleeps while the host makes the calls
HEADS, HEAD_DIM = 16, 72  # Gemma 4's tower (FlashInfer's head dim: 128, zero-padded)
PLAN_BYTES = 64 << 10  # more than a plan for one image of 2,520 patches writes (its tile indices and merge offsets)


def ahead(steps: list[list], sleeps: tuple[int, ...] = (SLEEP, 4 * SLEEP, 16 * SLEEP)) -> list:
    """The calls of `steps` (functions) made while the stream sleeps, a step's forward (a shorter sleep) queued after
    each step's calls; their results, once the device ran them. If the device caught up with the host, no plan was in
    flight: the calls are made again behind the next, longer sleep, as a loaded host can take longer than one. Fails
    if the device caught up behind every sleep."""
    for sleep in sleeps:
        torch.cuda.synchronize()
        torch.cuda._sleep(sleep)
        behind = torch.cuda.Event()
        behind.record()
        out = []
        for step in steps:
            out += [f() for f in step]
            torch.cuda._sleep(sleep // 10)
        caught_up = behind.query()
        torch.cuda.synchronize()
        if not caught_up:
            return out
    raise AssertionError(f"the device caught up with the host behind every sleep {sleeps}: no plan was in flight")


def attend(attn, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One encoder call's attention over one image (x: its q, k and v [3, n, heads, pad]): its plan, then one layer's
    run. Returns the output and the start of the page-locked buffer the plan wrote, which its copy reads."""
    n = x.shape[1]
    meta = SimpleNamespace(n=n, cu_host=[0, n], attn=None)
    attn.plan(meta)
    return attn.run(meta, x[0], x[1], x[2]), meta.attn._pin_memory_int_workspace_buffer[:PLAN_BYTES].clone()


@pytest.mark.parametrize("steps", [[[2520, 594, 594]], [[2520, 594], [594, 2520]]], ids=["one step", "two steps"])
def test_attention_planned_ahead_gives_each_image_its_own_output(steps) -> None:
    from llm_mini_infer.kernels.flashinfer import FlashInferVisionAttention

    dev = torch.device("cuda")
    attn = FlashInferVisionAttention(HEADS, HEAD_DIM, torch.bfloat16, dev)
    g = torch.Generator(device=dev).manual_seed(0)
    xs = [[F.pad(torch.randn((3, n, HEADS, HEAD_DIM), generator=g, device=dev).to(torch.bfloat16),
                 (0, attn.pad - HEAD_DIM)) for n in step] for step in steps]
    alone, written = [], {}
    for x in (x for step in xs for x in step):
        for w in attn.wrappers:  # zeros past the plan's bytes: only the bytes the plans write can differ
            w._pin_memory_int_workspace_buffer.zero_()
        out, written[x.shape[1]] = attend(attn, x)
        torch.cuda.synchronize()
        alone.append(out)
    assert not torch.equal(written[2520], written[594]), \
        "the two sizes' plans wrote the same bytes: the test could not tell them apart"
    got = ahead([[lambda x=x: attend(attn, x)[0] for x in step] for step in xs])
    sizes = [x.shape[1] for step in xs for x in step]
    for i, (a, b) in enumerate(zip(got, alone)):
        assert torch.equal(a, b), f"image {i} ({sizes[i]} patches) attended with another image's plan"


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from ..support.tiny_gemma import make_tiny_gemma_mm

    return make_tiny_gemma_mm(tmp_path_factory.mktemp("visionplans"), depth=2, gpu=True, seed=4)


@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "graphs"])
def test_encode_planned_ahead_gives_each_image_its_own_rows(tiny, graphs) -> None:
    """`ImageEncoder.encode` with one tower call per image, of images of 2,520, 594 and 2,268 patches (the tiny tower's
    patches are 4 pixels: at images.max_soft_tokens 280, the largest image's 280 soft tokens are a 42 x 60 grid):
    each image's encoder rows equal its rows encoded alone."""
    from llm_mini_infer.runner.cuda.images import ImageEncoder, VisionOptions

    from ..scripts.profile_tower import ImagesCall, capture

    dev = torch.device("cuda")
    enc = ImageEncoder(tiny, VisionOptions(encoder_rows=1024, encoder_budget=1024, max_soft_tokens=280), dev,
                       torch.bfloat16, buckets=[648, 2268, 2520])
    enc.allocate(enc.cfg.out_hidden, flashinfer=True)
    if graphs:
        capture(enc, enc.buckets)
    side = enc.cfg.merge * enc.cfg.patch  # one soft token's pixels
    gen = torch.Generator().manual_seed(0)
    pixels = [torch.randint(0, 256, (3, side * h, side * w), generator=gen, dtype=torch.uint8)
              for h, w in ((14, 20), (6, 11), (14, 18))]
    alone = []
    for px in pixels:
        one = ImagesCall(enc, [px])
        one(graphs)
        torch.cuda.synchronize()
        alone.append(one.rows(0))
    enc.per_image = True
    every = ImagesCall(enc, pixels)
    assert [h * w for _, h, w in every.grids] == [2520, 594, 2268]
    ahead([[lambda: every(graphs)]])
    for i, want in enumerate(alone):
        assert torch.equal(every.rows(i), want), f"image {i} ({every.grids[i]}) encoded with another image's plan"
