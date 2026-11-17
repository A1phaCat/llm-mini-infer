"""Tiny random-weight Gemma 4 checkpoints (transformers as the reference implementation), and a direct driver of the
model on the torch reference backends (CPU, no runner)."""

from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from llm_mini_infer.models.gemma4.backend import (
    Gemma4Context,
    Gemma4Meta,
    Gemma4Pools,
    TorchFullAttention,
    TorchSlidingAttention,
)
from llm_mini_infer.models.gemma4.model import FULL, SLIDING, Gemma4ForCausalLM, Rotary, build_model

from .paths import MODELS_DIR

GEMMA_TOKENIZER = MODELS_DIR / "gemma-4-26b-a4b"  # the 31B's, byte for byte
S, F = SLIDING, FULL
LAYERS = (S, S, F, S, S, F)  # the real pattern is 5 : 1; this keeps both kinds at depth with fewer layers


FP8_CPU_DIMS = dict(hidden_size=128, intermediate_size=256, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                    global_head_dim=128, num_global_key_value_heads=1)  # every projection dim a multiple of 128


def make_tiny_gemma(path: Path, *, vocab: int = 512, layers: tuple[str, ...] = LAYERS, window: int = 8,
                    seed: int = 0, scale: float = 0.05, prefix: str | None = None, k_eq_v: bool = True,
                    softcap: float | None = 30.0, gpu: bool = False, tokenizer: bool = False,
                    dims: dict | None = None) -> Path:
    """Save a random Gemma4ForCausalLM: sliding layers (head dim 32, 2 KV heads, a window of `window`) and full layers
    (head dim 64, 1 KV head, k_eq_v, proportional RoPE: 8 of 32 frequencies rotate), GQA, tied embeddings, final
    softcap. Every weight gets noise on top of HF's init, the norms and layer scalars away from 1. `prefix` renames
    `model.` to e.g. `model.language_model.` (the Gemma4ForConditionalGeneration layout), with config.json in that
    model's form (text_config with global_head_dim). `gpu`: dims our CUDA kernels and FlashInfer take (sliding head 256,
    full head 512). `tokenizer`: copy the real Gemma 4 tokenizer and chat template next to it."""
    from transformers import Gemma4ForCausalLM as HFGemma4
    from transformers import Gemma4TextConfig

    dims = dims or (dict(hidden_size=256, intermediate_size=512, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=256, global_head_dim=512, num_global_key_value_heads=1) if gpu else
                    dict(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=32, global_head_dim=64, num_global_key_value_heads=1))
    cfg = Gemma4TextConfig(
        vocab_size=vocab, num_hidden_layers=len(layers), layer_types=list(layers), attention_k_eq_v=k_eq_v,
        sliding_window=window, max_position_embeddings=4096, final_logit_softcapping=softcap,
        hidden_size_per_layer_input=0, tie_word_embeddings=True, eos_token_id=vocab - 1, rms_norm_eps=1e-6,
        rope_parameters={SLIDING: {"rope_type": "default", "rope_theta": 10_000.0},
                         FULL: {"rope_type": "proportional", "partial_rotary_factor": 0.25,
                                "rope_theta": 1_000_000.0}}, **dims)
    torch.manual_seed(seed)
    model = HFGemma4(cfg).float().eval()
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for _, p in model.named_parameters():
            p.add_(torch.randn(p.shape, generator=g) * (scale if p.dim() > 1 else 0.3))
        for name, b in model.named_buffers():
            if name.endswith("layer_scalar"):
                b.copy_(0.6 + 0.8 * torch.rand(b.shape, generator=g))
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    if prefix:
        from safetensors.torch import load_file, save_file

        f = path / "model.safetensors"
        sd = load_file(f)
        save_file({(prefix + k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}, f)
        conf = json.loads((path / "config.json").read_text())
        plc = conf.pop("per_layer_config")
        over = next(iter(plc.values()))
        conf.update(global_head_dim=over["head_dim"], num_global_key_value_heads=over["num_key_value_heads"])
        (path / "config.json").write_text(json.dumps({
            "architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4", "text_config": conf,
            "vision_config": None, "eos_token_id": [1, vocab - 1], "tie_word_embeddings": True}, indent=1))
    if tokenizer:
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            shutil.copy(GEMMA_TOKENIZER / name, path / name)
    return path


ASSISTANT_LAYERS = (S, S, S, F)  # the 31B's assistant: 3 sliding, 1 full


def make_tiny_assistant(path: Path, target_dir: Path, *, hidden: int = 32, intermediate: int = 64,
                        layers: tuple[str, ...] = ASSISTANT_LAYERS, seed: int = 0, scale: float = 0.08) -> Path:
    """Save a random `Gemma4AssistantForCausalLM` for the tiny target in `target_dir`: its layers Q-only over the
    target's KV (`num_kv_shared_layers` = all), the target's head dims, window, RoPE and query heads, a backbone of the
    target's width, the tied head without softcap, no centroid head (as the 31B's assistant). Noise on every weight,
    the norms and layer scalars away from 1."""
    from transformers import Gemma4AssistantConfig, Gemma4AssistantForCausalLM, Gemma4TextConfig

    t = json.loads((target_dir / "config.json").read_text())
    t = t.get("text_config", t)
    plc = next(iter((t.get("per_layer_config") or {}).values()), {})
    tc = Gemma4TextConfig(
        vocab_size=t["vocab_size"], num_hidden_layers=len(layers), layer_types=list(layers), attention_k_eq_v=True,
        sliding_window=t["sliding_window"], hidden_size=hidden, intermediate_size=intermediate,
        num_attention_heads=t["num_attention_heads"], num_key_value_heads=t["num_key_value_heads"],
        head_dim=t["head_dim"], global_head_dim=t.get("global_head_dim") or plc.get("head_dim"),
        num_global_key_value_heads=t.get("num_global_key_value_heads") or plc.get("num_key_value_heads"),
        hidden_size_per_layer_input=0, vocab_size_per_layer_input=0, num_kv_shared_layers=len(layers),
        tie_word_embeddings=True, final_logit_softcapping=None, rms_norm_eps=1e-6, eos_token_id=t.get("eos_token_id"),
        max_position_embeddings=4096, rope_parameters=t["rope_parameters"])
    cfg = Gemma4AssistantConfig(text_config=tc, backbone_hidden_size=t["hidden_size"], use_ordered_embeddings=False)
    torch.manual_seed(seed)
    model = Gemma4AssistantForCausalLM(cfg).float().eval()
    g = torch.Generator().manual_seed(seed + 7)
    with torch.no_grad():
        for _, p in model.named_parameters():
            p.add_(torch.randn(p.shape, generator=g) * (scale if p.dim() > 1 else 0.3))
        for name, b in model.named_buffers():
            if name.endswith("layer_scalar"):
                b.copy_(0.6 + 0.8 * torch.rand(b.shape, generator=g))
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    return path


def hf_assistant(model_dir: Path, dtype: torch.dtype = torch.float32):
    from transformers import Gemma4AssistantForCausalLM

    return Gemma4AssistantForCausalLM.from_pretrained(model_dir, dtype=dtype).eval()


# RedHatAI/gemma-4-31B-it-FP8-block's quantization_config (config.json), as published
REDHAT_FP8_BLOCK = {
    "config_groups": {"FP8_BLOCK": {
        "format": "float-quantized",
        "input_activations": {"actorder": None, "block_structure": None, "dynamic": True, "group_size": 128,
                              "num_bits": 8, "observer": None, "observer_kwargs": {}, "scale_dtype": None,
                              "strategy": "group", "symmetric": True, "type": "float", "zp_dtype": None},
        "output_activations": None, "targets": ["Linear"],
        "weights": {"actorder": None, "block_structure": [128, 128], "dynamic": False, "group_size": None,
                    "num_bits": 8, "observer": "memoryless_minmax", "observer_kwargs": {}, "scale_dtype": None,
                    "strategy": "block", "symmetric": True, "type": "float", "zp_dtype": None}}},
    "format": "float-quantized", "global_compression_ratio": None,
    "ignore": ["re:.*vision.*", "lm_head", "re:.*embed_tokens.*"], "kv_cache_scheme": None,
    "quant_method": "compressed-tensors", "quantization_status": "compressed", "sparsity_config": {},
    "transform_config": {}, "version": "0.14.0.1"}
FP8_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def quantize_block_bf16_scale(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """llm-compressor's FP8_BLOCK as RedHat's checkpoint stores it: per 128 x 128 block the scale bf16(max |w| / 448)
    (BF16, as in that checkpoint: each of its scales is exactly that), the weight e4m3(w / scale), saturated."""
    from llm_mini_infer.kernels import fp8

    n, k = w.shape
    wb = w.float().reshape(n // 128, 128, k // 128, 128)
    s = (wb.abs().amax(dim=(1, 3)) / 448).to(torch.bfloat16)
    q = (wb / s.float()[:, None, :, None]).clamp(-448, 448).to(fp8.FP8).reshape(n, k)
    return q, s


def make_tiny_gemma_fp8(path: Path, *, seed: int = 0, gpu: bool = False, scale: float = 0.05,
                        layers: tuple[str, ...] = LAYERS, qk_norm: float = 0.25) -> tuple[Path, Path]:
    """A random Gemma 4 in compressed-tensors FP8_BLOCK, RedHat's format: every text projection e4m3 with a BF16
    `weight_scale` per 128 x 128 block, the embeddings BF16, `quantization_config` the published one
    (`REDHAT_FP8_BLOCK`), the Gemma4ForConditionalGeneration layout (`model.language_model.*`; the full layers k_eq_v,
    without v_proj). Dims: `FP8_CPU_DIMS`, or with `gpu` those our CUDA kernels take. The q and k norms' weights are
    scaled by `qk_norm`, as a trained model keeps its attention logits moderate at scale 1.0 (near 1 the random
    model's FP8 noise moves its logits by 42% RMS, at a quarter by 7%; tests/models/test_gemma4_gpu.py's fixture does
    the same for BF16). Returns (the FP8 checkpoint, the same model with its weights dequantized, in fp32 and the
    Gemma4ForCausalLM layout): transformers runs the latter as the reference."""
    from safetensors.torch import load_file, save_file

    from llm_mini_infer.kernels import fp8

    deq_dir, fp8_dir = path / "dequantized", path / "fp8"
    make_tiny_gemma(deq_dir, layers=layers, seed=seed, scale=scale, gpu=gpu, dims=None if gpu else FP8_CPU_DIMS)
    make_tiny_gemma(fp8_dir, layers=layers, seed=seed, scale=scale, gpu=gpu, dims=None if gpu else FP8_CPU_DIMS,
                    prefix="model.language_model.")
    sd = load_file(deq_dir / "model.safetensors")
    sd = {k: v * qk_norm if k.endswith(("q_norm.weight", "k_norm.weight")) else v for k, v in sd.items()}
    quantized, deq = {}, {}
    for k, v in sd.items():
        parts = k.split(".")
        if k.endswith(".weight") and len(parts) > 2 and parts[-2] in FP8_PROJ:
            q, sc = quantize_block_bf16_scale(v)
            name = "model.language_model." + k[len("model."):]
            quantized[name], quantized[name + "_scale"] = q, sc
            deq[k] = fp8.dequant_weight(q, sc.float())
        else:  # the same tensor in both (embeddings in BF16, as the checkpoint stores them)
            deq[k] = v.to(torch.bfloat16) if v.dim() > 1 else v
            quantized[("model.language_model." + k[len("model."):]) if k.startswith("model.") else k] = deq[k]
    save_file(deq, deq_dir / "model.safetensors")
    save_file(quantized, fp8_dir / "model.safetensors")
    conf = json.loads((fp8_dir / "config.json").read_text())
    conf["quantization_config"] = REDHAT_FP8_BLOCK
    (fp8_dir / "config.json").write_text(json.dumps(conf, indent=1))
    return fp8_dir, deq_dir


MM_IMAGE, MM_BOI, MM_EOI = 500, 501, 502  # the tiny multimodal model's image, begin-of-image and end-of-image tokens
MM_PATCH, MM_SOFT = 4, 70  # its patches of 4 x 4 pixels, at most 70 soft tokens an image (630 patches)


def make_tiny_gemma_mm(path: Path, *, window: int = 8, seed: int = 0, scale: float = 0.05, depth: int = 2,
                       bidirectional: str | None = "vision", vision_scale: float = 0.08, gpu: bool = False,
                       dims: dict | None = None) -> Path:
    """Save a random `Gemma4ForConditionalGeneration`: `make_tiny_gemma`'s text model (vocab 512, layers S S F S S F,
    window `window`; `gpu`: the dims our CUDA kernels and FlashInfer take, as `make_tiny_gemma`'s; `dims`: others) with
    `use_bidirectional_attention` = `bidirectional`, and a vision tower of `depth` layers (hidden
    32, 2 heads of 16, MLP 64, patches of 4 pixels, 3 x 3 pooling, position tables of 256: a strip of 70 soft tokens
    is 210 patches wide), standardised, noise on every weight, the norms away from 1, random standardisation. Images
    take at most `MM_SOFT` soft tokens (processor_config.json); the image tokens are `MM_IMAGE`, `MM_BOI`,
    `MM_EOI`."""
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig, Gemma4VisionConfig

    tc = Gemma4TextConfig(
        vocab_size=512, num_hidden_layers=len(LAYERS), layer_types=list(LAYERS), attention_k_eq_v=True,
        sliding_window=window, max_position_embeddings=4096, final_logit_softcapping=30.0,
        hidden_size_per_layer_input=0, tie_word_embeddings=True, eos_token_id=511, rms_norm_eps=1e-6,
        use_bidirectional_attention=bidirectional, pad_token_id=0,
        rope_parameters={SLIDING: {"rope_type": "default", "rope_theta": 10_000.0},
                         FULL: {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1_000_000.0}},
        **(dims or (dict(hidden_size=256, intermediate_size=512, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=256, global_head_dim=512, num_global_key_value_heads=1) if gpu else
                    dict(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=32, global_head_dim=64, num_global_key_value_heads=1))))
    vc = Gemma4VisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=depth, num_attention_heads=2,
                            num_key_value_heads=2, head_dim=16, patch_size=MM_PATCH, pooling_kernel_size=3,
                            position_embedding_size=256, standardize=True, rms_norm_eps=1e-6,
                            rope_parameters={"rope_type": "default", "rope_theta": 100.0})
    cfg = Gemma4Config(text_config=tc, vision_config=vc, audio_config=None, image_token_id=MM_IMAGE,
                       boi_token_id=MM_BOI, eoi_token_id=MM_EOI, video_token_id=503, tie_word_embeddings=True)
    torch.manual_seed(seed)
    model = Gemma4ForConditionalGeneration(cfg).float().eval()
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            s = vision_scale if "vision" in name else scale
            p.add_(torch.randn(p.shape, generator=g) * (s if p.dim() > 1 else 0.3))
        for name, b in model.named_buffers():
            if name.endswith("layer_scalar"):
                b.copy_(0.6 + 0.8 * torch.rand(b.shape, generator=g))
            elif name.endswith("std_bias"):
                b.copy_(torch.randn(b.shape, generator=g) * 0.5)
            elif name.endswith("std_scale"):
                b.copy_(0.5 + torch.rand(b.shape, generator=g))
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    (path / "processor_config.json").write_text(json.dumps({"image_processor": {
        "image_processor_type": "Gemma4ImageProcessor", "do_convert_rgb": True, "do_normalize": False,
        "do_rescale": True, "do_resize": True, "image_mean": [0.0, 0.0, 0.0], "image_std": [1.0, 1.0, 1.0],
        "max_soft_tokens": MM_SOFT, "patch_size": MM_PATCH, "pooling_kernel_size": 3, "resample": 3,
        "rescale_factor": 1 / 255}}))
    return path


def make_tiny_gemma_mm_fp8(path: Path, *, window: int = 8, seed: int = 0, depth: int = 2, gpu: bool = False,
                           qk_norm: float = 0.25) -> Path:
    """`make_tiny_gemma_mm`'s model in compressed-tensors FP8_BLOCK, as `make_tiny_gemma_fp8`'s: every text projection
    e4m3 with a BF16 `weight_scale` per 128 x 128 block, the embeddings and the vision tower BF16 (the published
    checkpoint's `ignore`), `quantization_config` the published one (`REDHAT_FP8_BLOCK`). Dims: `FP8_CPU_DIMS`, or
    with `gpu` those our CUDA kernels take. The text model's q and k norms' weights are scaled by `qk_norm`, as
    `make_tiny_gemma_fp8`'s are."""
    from safetensors.torch import load_file, save_file

    make_tiny_gemma_mm(path, window=window, seed=seed, depth=depth, gpu=gpu, dims=None if gpu else FP8_CPU_DIMS)
    f = path / "model.safetensors"
    out = {}
    for k, v in load_file(f).items():
        text = k.startswith("model.language_model.")
        if text and k.endswith(("q_norm.weight", "k_norm.weight")):
            v = v * qk_norm
        if text and k.endswith(".weight") and k.split(".")[-2] in FP8_PROJ:
            out[k], out[k + "_scale"] = quantize_block_bf16_scale(v)
        else:
            out[k] = v.to(torch.bfloat16) if v.dim() > 1 else v
    save_file(out, f)
    conf = json.loads((path / "config.json").read_text())
    conf["quantization_config"] = REDHAT_FP8_BLOCK
    (path / "config.json").write_text(json.dumps(conf, indent=1))
    return path


def hf_mm_model(model_dir: Path, dtype: torch.dtype = torch.float32):
    from transformers import Gemma4ForConditionalGeneration

    return Gemma4ForConditionalGeneration.from_pretrained(model_dir, dtype=dtype).eval()


def hf_image_processor(model_dir: Path):
    from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor

    ip = json.loads((model_dir / "processor_config.json").read_text())["image_processor"]
    return Gemma4ImageProcessor(patch_size=ip["patch_size"], max_soft_tokens=ip["max_soft_tokens"],
                                pooling_kernel_size=ip["pooling_kernel_size"])


def hf_mm_logits(model_dir: Path, ids: list[int], images: list, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """transformers' logits [T, V] (fp32) for a prompt whose image spans (runs of `MM_IMAGE`, each its image's soft
    tokens) take `images` (PIL, in order), with the bidirectional masks HF builds from `mm_token_type_ids`: one
    forward over the whole sequence."""
    model = hf_mm_model(model_dir, dtype)
    ids_t = torch.tensor([ids])
    kw = {}
    if images:
        out = hf_image_processor(model_dir)(images=images, return_tensors="pt")
        kw = dict(pixel_values=out["pixel_values"].to(dtype), image_position_ids=out["image_position_ids"],
                  mm_token_type_ids=(ids_t == MM_IMAGE).long())
    with torch.no_grad():
        return model(input_ids=ids_t, **kw).logits[0].float()


def hf_model(model_dir: Path, dtype: torch.dtype = torch.float32):
    from transformers import Gemma4ForCausalLM as HFGemma4

    return HFGemma4.from_pretrained(model_dir, dtype=dtype).eval()


def hf_logits(model_dir: Path, ids: list[int], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Reference logits [T, V] (fp32) from transformers' implementation, one forward over the whole sequence."""
    model = hf_model(model_dir, dtype)
    with torch.no_grad():
        return model(torch.tensor([ids])).logits[0].float()


class Direct:
    """Our model on the torch reference backends, one forward over whole sequences from position 0 (each its own
    pages in both pools): `forward(seqs)` returns the final hidden states of every token and, per layer, the residual
    stream after it (for layer-by-layer comparisons)."""

    def __init__(self, model_dir: Path, page_size: int = 4, pages: int = 256, dtype: torch.dtype = torch.float32):
        self.dev = torch.device("cpu")
        self.model: Gemma4ForCausalLM = build_model(model_dir, self.dev, dtype)
        cfg = self.cfg = self.model.cfg
        self.ps = page_size
        ns, nf = len(cfg.sliding_layers), len(cfg.full_layers)
        wkv = torch.zeros((ns, pages, 2, page_size, cfg.sliding_kv_heads, cfg.sliding_head_dim), dtype=dtype)
        fshape = (nf, pages, page_size, cfg.full_kv_heads, cfg.full_head_dim) if cfg.k_eq_v else \
            (nf, pages, 2, page_size, cfg.full_kv_heads, cfg.full_head_dim)
        self.pools = Gemma4Pools(wkv=wkv, fkv=torch.zeros(fshape, dtype=dtype), page_size=page_size,
                                 window=cfg.sliding_window)
        self.rotary = Rotary(cfg, 4096, self.dev, dtype)
        self.sliding = TorchSlidingAttention(self.pools)
        self.full = TorchFullAttention(self.pools, self.rotary, cfg.k_eq_v)

    def forward(self, seqs: list[list[int]]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        ps = self.ps
        n = len(seqs)
        lens = np.asarray([len(s) for s in seqs], np.int64)
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        max_pages = int(-(-lens.max() // ps))
        table = torch.zeros((n, max_pages), dtype=torch.int32)
        page = 1
        for r, ln in enumerate(lens.tolist()):
            k = -(-ln // ps)
            table[r, :k] = torch.arange(page, page + k, dtype=torch.int32)
            page += k
        pos = torch.cat([torch.arange(ln) for ln in lens.tolist()])
        row = torch.repeat_interleave(torch.arange(n), torch.as_tensor(lens))
        pg = table[row, pos // ps].long()
        meta = Gemma4Meta(num_tokens=int(lens.sum()), positions=pos, f_page=pg, f_off=pos % ps, w_page=pg,
                          w_off=pos % ps, n_rows=n, row_start_host=starts, row_nq_host=lens, row_kv_len_host=lens,
                          row_slot_host=np.arange(n), row_wlo_host=np.zeros(n, np.int64), page_table=table,
                          wpage_table=table)
        self.table = table  # each sequence's pages in both pools (test_gemma4_assistant's drafter reads them)
        ctx = Gemma4Context(meta, self.sliding, self.full, self.rotary, self.pools)
        ids = torch.as_tensor(np.concatenate([np.asarray(s) for s in seqs]), dtype=torch.long)
        per_layer: list[torch.Tensor] = []
        m = self.model
        handles = []
        from llm_mini_infer.kernels import gemma_ops

        orig = gemma_ops.sandwich
        count = [0]

        def spy(*a, **kw):  # the residual after each layer's MLP sandwich (every second call)
            r, x = orig(*a, **kw)
            count[0] += 1
            if count[0] % 2 == 0:
                per_layer.append(r.clone())
            return r, x

        gemma_ops.sandwich = spy
        try:
            hidden = m(ids, ctx)
        finally:
            gemma_ops.sandwich = orig
            for h in handles:
                h.remove()
        return hidden, per_layer

    def logits(self, seqs: list[list[int]]) -> torch.Tensor:
        hidden, _ = self.forward(seqs)
        return self.model.logits(hidden)


# -- driving GemmaRunner by hand (CPU) -------------------------------------------------------------------------------


GREEDY = (0.0, 1.0, 0, 0.0, 1)


TINY_FULL_RATIO = 4  # both tiny models: a full page holds 4 window pages' positions (`runner.cuda.gemma._Arena`)


def gemma_pools(max_num_seqs: int = 4, budget: int = 256, page_size: int = 4, num_pages: int = 512,
                max_model_len: int = 1024, full: int = 0, mtp_k: int = 0):
    """One KV arena: window pages of `page_size` positions, full pages of `full` (default: the tiny models' ratio),
    `num_pages` pages of either kind."""
    from llm_mini_infer.runner.base import PoolConfig

    return PoolConfig(page_size=full or TINY_FULL_RATIO * page_size, num_pages=num_pages,
                      num_state_slots=max_num_seqs + 1, max_num_seqs=max_num_seqs, max_num_batched_tokens=budget,
                      max_model_len=max_model_len, eos_token_ids=(), mtp_k=mtp_k, window_page_size=page_size)


def gemma_cpu_runner(model_dir: Path, pc, **kw):
    from llm_mini_infer.runner.cuda.gemma import GemmaRunner

    r = GemmaRunner(model_dir, device="cpu", dtype=torch.float32, max_num_seqs=pc.max_num_seqs,
                    max_num_batched_tokens=pc.max_num_batched_tokens, max_model_len=pc.max_model_len, check=True,
                    **kw)
    r.configure(pc)
    return r


class GDriver:
    """Plans by hand for a GemmaRunner (both kinds of page, one arena), in the engine's call order, collecting the
    logits of every computed position through the logits hook, keyed by (slot, position). Rows: (slot, tokens or None
    for device input, pos0, sample). `release`: drop each slot's window pages below its row's window (as the scheduler
    does), back to the arena's free list, from which the next pages of either kind come, so a read of a released page
    (or of a page reused as the other kind) shows."""

    def __init__(self, runner, pc, window: int, release: bool = True):
        from llm_mini_infer.sched.plan import PlanBuilder

        self.r, self.pc, self.W = runner, pc, window
        self.fps, self.wps = pc.page_size, pc.window_page_size
        self.b = PlanBuilder(pc.max_num_seqs, pc.max_num_batched_tokens, pc.page_size, pc.max_model_len, 0, window,
                             pc.window_page_size)
        self.step_id = 0
        self.fpages: dict[int, list[int]] = {}
        self.wpages: dict[int, list[int]] = {}
        self.free = list(range(pc.num_pages - 1, 0, -1))  # the arena's, for both kinds
        self.release = release
        self.refs: dict[int, int] = {}  # pages two slots share (a resume) are freed by their last holder
        self.logits: dict[tuple[int, int], torch.Tensor] = {}
        self._slots: list[int] = []
        self.reused = 0  # pages taken from the free list after an earlier holder gave them back
        self.switched = 0  # of which held the other kind before (a window page reused as a full page, or back)
        self._kind: dict[int, str] = {}
        self._given: set[int] = set()
        runner.logits_hook = self._hook

    def _hook(self, chunk) -> None:
        lg = chunk.logits.detach().to("cpu", torch.float32)
        for i, (row, pos) in enumerate(zip(chunk.rows.tolist(), chunk.positions.tolist())):
            self.logits[(self._slots[row], pos)] = lg[i]

    def _take(self, kind: str) -> int:
        p = self.free.pop()
        self.reused += p in self._given
        self.switched += self._kind.get(p, kind) != kind
        self._kind[p] = kind
        self.refs[p] = 1
        return p

    def _cover(self, slot: int, upto: int) -> None:
        for kind, pages, ps, add in (("full", self.fpages, self.fps, self.b.add_pages),
                                     ("window", self.wpages, self.wps, self.b.add_wpages)):
            have = pages.setdefault(slot, [])
            new = [self._take(kind) for _ in range(max(0, -(-upto // ps) - len(have)))]
            if new:
                add(slot, len(have), new)
                have.extend(new)

    def _drop(self, p: int) -> None:
        self.refs[p] -= 1
        if not self.refs[p]:
            self.free.append(p)
            self._given.add(p)

    def _slide(self, slot: int, q: int) -> None:
        lo = max(0, q - self.W + 1) // self.wps
        have = self.wpages.get(slot, [])
        old = [(i, p) for i, p in enumerate(have[:lo]) if p]
        if old and self.release:
            self.b.add_wpages(slot, old[0][0], [0] * (lo - old[0][0]))
            for i, p in old:
                have[i] = 0
                self._drop(p)

    def plan(self, rows, admit=(), free=(), seq_len=None, share=None):
        from llm_mini_infer.sched.plan import PRECISE, SAMPLE, STATE_ZERO

        b = self.b
        b.begin()
        for s in free:
            b.add_free(s)
            for p in self.fpages.pop(s, []) + self.wpages.pop(s, []):
                if p:
                    self._drop(p)
        for slot, (temp, top_p, top_k, min_p, seed) in admit:
            b.add_admit(slot, slot, (seq_len or {}).get(slot, 0), temp, top_p, top_k, min_p, seed)
        for slot, (src, upto) in (share or {}).items():  # share src's pages below `upto`, copy a partial one
            for kind, pages, ps, add, copy in (("full", self.fpages, self.fps, b.add_pages, b.add_kv_copy),
                                               ("window", self.wpages, self.wps, b.add_wpages, b.add_wkv_copy)):
                whole = upto // ps
                mine = list(pages[src][:whole])
                for p in mine:
                    if p:
                        self.refs[p] += 1
                if any(mine):
                    first = next(i for i, p in enumerate(mine) if p)
                    add(slot, first, mine[first:])
                if upto % ps:
                    own = self._take(kind)
                    add(slot, whole, [own])
                    copy(pages[src][whole], own)
                    mine.append(own)
                pages[slot] = mine
        dev = [r for r in rows if r[1] is None]
        self._slots = [r[0] for r in dev]
        for slot, _, pos0, *_ in dev:
            self._slide(slot, pos0)
            self._cover(slot, pos0 + 2)
        if dev:
            b.add_device_rows(np.asarray([r[0] for r in dev], np.int32))
        for slot, toks, pos0, smp, *precise in rows:  # (slot, tokens, pos0, sample[, PRECISE])
            if toks is None:
                continue
            self._slide(slot, pos0)
            self._cover(slot, pos0 + len(toks) + (2 if smp else 0))
            b.add_explicit_row(slot, np.asarray(toks, np.int32), pos0,
                               (STATE_ZERO if pos0 == 0 else 0) | (SAMPLE if smp else 0) |
                               (PRECISE if precise and precise[0] else 0))
            self._slots.append(slot)
        self.step_id += 1
        return b.finish(self.step_id)

    def run(self, steps, masked: bool = False):
        """steps: dicts of plan() kwargs; returns the StepResults. `masked`: every SAMPLE row is MASKED, under a mask
        that allows every token (the grammar split path: the forward in launch_forward, the sampling in
        launch_sample)."""
        from llm_mini_infer.sched.plan import MASKED, SAMPLE

        out, prev = [], None
        for s in steps:
            plan = self.plan(**s)
            n = 0
            if masked:
                sample = (plan.sched_flags & SAMPLE) != 0
                plan.sched_flags[sample] |= MASKED
                n = int(sample.sum())
            h = self.r.launch_forward(plan)
            if prev is not None:
                out.append(self.r.wait(prev))
            if n:
                assert h.split is not None, "a step with masks splits"
                self.r.mask_buffer(h)[:n] = -1
            self.r.launch_sample(h, n)
            prev = h
        if prev is not None:
            out.append(self.r.wait(prev))
        return out


@contextmanager
def poisoned_rope(runner, pos: int):
    """The sliding layers' RoPE cos at `pos` NaN in a GemmaRunner's table (in place: its CUDA graphs read it too),
    restored on exit: q and K of a token there go NaN in every sliding layer, so the row that computes it writes NaN
    KV, and every later position of its sequence reads it (its full layers read every earlier position)."""
    cos = runner.rotary.cos[S]
    with torch.inference_mode():  # the runner's tables are inference tensors
        keep = cos[pos].clone()
        cos[pos] = float("nan")
    try:
        yield
    finally:
        with torch.inference_mode():
            cos[pos] = keep


def spy_modes(runner) -> list[str]:
    """The body each later step of a GemmaRunner runs, in order ("eager", "decode" or "break": its `_mode`)."""
    seen: list[str] = []
    mode = runner._mode

    def spy(lay, split=False):
        out = mode(lay, split)
        seen.append(out[0])
        return out

    runner._mode = spy
    return seen
