"""Images through the CUDA runner on CPU (torch reference kernels, fp32), the real scheduler and the engine loop,
against transformers' `Qwen3_5ForConditionalGeneration` on a tiny random multimodal model.

Teacher-forced logits of prompts with images match HF's forward (pixel_values from the same pixels, HF's own M-RoPE
positions), whole or chunked across a span; greedy decoding continues HF's sequence (the M-RoPE offset after the
image); MTP's outputs equal MTP off's; a cache hit inside an image span gives the cold run's tokens.
"""

from __future__ import annotations

import pytest
import torch

from ..support.qwen_images import PAD, Engine, Img, prompt_of, text
from ..support.qwen_images import vl as vl  # the fixture

TOL = 1e-4  # fp32 on both sides; the patch embedding is a GEMM here, a Conv3d in HF


def hf_logits(model, ids: list[int], spans) -> torch.Tensor:
    from llm_mini_infer.models.qwen3_5.vision import QwenVisionEncoder

    t = torch.tensor([ids])
    kw = {}
    if spans:
        patches = torch.cat([QwenVisionEncoder.patches(_enc_stub(), im.px) for _, im in spans])
        kw = dict(pixel_values=patches, image_grid_thw=torch.tensor([[1, im.gh, im.gw] for _, im in spans]),
                  mm_token_type_ids=(t == PAD).int())
    with torch.no_grad():
        return model(input_ids=t, **kw).logits[0].float()


def _enc_stub():
    """`QwenVisionEncoder.patches` needs only the config and the normalisation buffers."""
    from llm_mini_infer.models.qwen3_5.vision import QwenVisionEncoder, VisionConfig, vision_buffers

    cfg = VisionConfig(depth=0, hidden=64, heads=4, intermediate=96, patch=16, temporal=2, merge=2, out_hidden=64,
                       num_pos=64, in_channels=3, act="gelu_pytorch_tanh", rope_theta=10000.0, mean=(0.5,) * 3,
                       std=(0.5,) * 3, rescale=1 / 255)
    e = QwenVisionEncoder.__new__(QwenVisionEncoder)
    e.cfg = cfg
    for k, v in vision_buffers(cfg).items():
        setattr(e, k, v)
    return e


@pytest.mark.parametrize("budget", [128, 8], ids=["one-step", "chunked"])
def test_prompt_logits_match_transformers(vl, tmp_path, budget) -> None:
    """One image (a 4 x 6 patch grid: 6 tokens) between text, prefilled in one step or in 8-token chunks that split
    the span: every prompt position's logits match HF's forward."""
    path, model = vl
    ids, spans = prompt_of([text(9), Img(4, 6, 1), text(7, 1)])
    e = Engine(path, tmp_path, budget=budget)
    e.add(0, ids, spans, max_tokens=1)
    e.run()
    ref = hf_logits(model, ids, spans)
    ours = e.prompt_logits(0, len(ids))
    assert (ours - ref).abs().max().item() < TOL
    assert e.out[0] == [int(ref[-1].argmax())]
    assert e.sched.encoder_images == 1


def test_two_images_and_greedy_decoding_continue_transformers(vl, tmp_path) -> None:
    """Two images (one twice as wide) and text between: the prompt matches HF, and 8 greedy tokens continue HF's
    sequence (their RoPE positions carry the offset after the last image)."""
    path, model = vl
    ids, spans = prompt_of([text(5, 2), Img(4, 4, 3), text(3, 3), Img(2, 8, 4), text(6, 4)])
    e = Engine(path, tmp_path, budget=64)
    e.add(0, ids, spans, max_tokens=8)
    e.run()
    gen = e.out[0]
    ref = hf_logits(model, ids + gen, spans)
    assert (e.prompt_logits(0, len(ids)) - ref[:len(ids)]).abs().max().item() < TOL
    assert [int(x) for x in ref[len(ids) - 1:-1].argmax(-1)] == gen


def test_mtp_outputs_equal_mtp_off(vl, tmp_path) -> None:
    """With MTP (k = 2) the drafter reads the image's rows at image positions and the verification keeps exactly the
    target's tokens: greedy outputs equal MTP off's, also when 8-token chunks end inside or right before the span."""
    path, _ = vl
    ids, spans = prompt_of([text(6, 5), Img(4, 6, 5), text(5, 6)])
    outs = []
    for k, budget in ((0, 128), (2, 128), (2, 8)):
        e = Engine(path, tmp_path, budget=budget, k=k)
        e.add(0, ids, spans, max_tokens=12)
        e.run()
        outs.append(e.out[0])
    assert outs[0] == outs[1] == outs[2]


def test_a_hit_inside_the_span_gives_the_cold_tokens(vl, tmp_path) -> None:
    """Three prompts with the same text and image and different text after it, 8-token chunks, the prefix cache on.
    The first leaves checkpoints at its chunk ends (8, 16, 24; the span is [5, 21)); the others resume at 16, inside
    the span, and read the image's cached embeddings for the rest of it. Every request's tokens equal the cache-off
    run's."""
    path, _ = vl
    img = Img(8, 8, 6)
    prompts = [prompt_of([text(4, 7), img, text(3 + i, 8 + i)]) for i in range(3)]
    outs = []
    for caching in (False, True):
        e = Engine(path, tmp_path, budget=8, caching=caching)
        for rid, (ids, spans) in enumerate(prompts):
            e.add(rid, ids, spans, max_tokens=6)
            e.run()
        outs.append(e.out)
    assert outs[0] == outs[1]
    assert e.sched.prefix_hits == 2 * 16 and e.sched.encoder_hits == 2 and e.sched.encoder_images == 1


def test_gemma_buckets_follow_max_soft_tokens() -> None:
    """Gemma 4's tower buckets at a 2,048-token encoder budget follow images.max_soft_tokens: dense every 128 patches
    to the largest image (at least 2,048), then about 1.25x apart to the budget's 18,432 patches; 70 drops the 280
    list's 2,178, 2,304 and 2,439. A 400 x 225 image at 70 (594 patches) takes the 648 bucket."""
    from llm_mini_infer.runner.cuda.images import family_buckets
    from llm_mini_infer.vision.gemma import GemmaVision

    def buckets(k: int) -> list[int]:
        return family_buckets(GemmaVision(boi_id=255999, image_id=258880, eoi_id=258882, max_soft_tokens=k), 3, 2048)

    at70 = [261, 387, 513, 648, 774, 900, 1026, 1152, 1287, 1413, 1539, 1665, 1800, 1926, 2052, 2565, 3204, 4104,
            5121, 6408, 8064, 10116, 12672, 15876, 18432]
    assert buckets(70) == at70
    assert sorted(set(buckets(280)) - set(at70)) == [2178, 2304, 2439] and set(at70) <= set(buckets(280))
    assert next(b for b in at70 if b >= 594) == 648
