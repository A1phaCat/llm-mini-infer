"""Tiny random-weight Qwen3.5 checkpoints (transformers as the reference implementation) and a plan driver for the
CUDA runner, on CPU or (`gpu` dims, `cuda_runner`) on CUDA."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from llm_mini_infer.runner.base import PoolConfig, StepResult
from llm_mini_infer.runner.cuda.runner import CudaRunner, LogitsChunk
from llm_mini_infer.sched.plan import PRECISE, SAMPLE, STATE_ZERO, PlanBuilder, StepPlan

LAYERS = ("linear_attention", "full_attention", "linear_attention", "full_attention")
CPU_DIMS = dict(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                linear_key_head_dim=16, linear_value_head_dim=16)
# Dims our CUDA kernels take, so a CUDA runner captures graphs: FlashInfer's attention (heads of 64) and GDN prefill
# (key and value heads of 128), our GDN decode and MTP kernels (128 × 128 states); smaller ones run the torch references
GPU_DIMS = dict(hidden_size=128, intermediate_size=256, num_attention_heads=2, num_key_value_heads=2, head_dim=64,
                linear_key_head_dim=128, linear_value_head_dim=128)


def add_mtp_layer(path: Path, model, seed: int, scale: float, fp8: bool = False) -> None:
    """Add a random MTP layer under the checkpoint's names (`mtp.fc`, `mtp.pre_fc_norm_*`, `mtp.layers.0.*` as a
    full-attention decoder layer, `mtp.norm`) and `mtp_num_hidden_layers: 1` to config.json. It is built from a copy
    of the model's first full-attention layer plus noise, so it is a plausible layer of the same shapes."""
    from safetensors.torch import load_file, save_file

    from llm_mini_infer.kernels import fp8 as f8

    attn = next(i for i, t in enumerate(model.config.layer_types) if t == "full_attention")
    g = torch.Generator().manual_seed(seed + 77)
    h = model.config.hidden_size
    sd = load_file(path / "model.safetensors")
    pre = next(k for k in sd if k.endswith("embed_tokens.weight")).removesuffix("embed_tokens.weight")
    lay = f"{pre}layers.{attn}."
    dt = next(v.dtype for k, v in sd.items()
              if v.dtype != torch.float8_e4m3fn and v.dim() > 1 and not k.endswith("_scale_inv"))
    mtp: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith(lay) and not k.endswith("_scale_inv"):
            if v.dtype == torch.float8_e4m3fn:  # a block-FP8 checkpoint's projection: its values (`fp8` requantizes)
                v = f8.dequant_weight(v, sd[k + "_scale_inv"]).to(dt)
            mtp["mtp.layers.0." + k[len(lay):]] = v.clone()
    for k, v in list(mtp.items()):
        mtp[k] = (v.float() + torch.randn(v.shape, generator=g) * (scale if v.dim() > 1 else 0.3)).to(v.dtype)
    mtp["mtp.fc.weight"] = (torch.randn((h, 2 * h), generator=g) * (2 * h) ** -0.5).to(dt)
    for n in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"):
        mtp[f"mtp.{n}.weight"] = (torch.randn(h, generator=g) * 0.3).to(dt)
    if fp8:
        for k in [k for k in mtp if k.endswith(".weight") and k.split(".")[-2] in FP8_PROJECTIONS]:
            mtp[k], mtp[k + "_scale_inv"] = f8.quantize_weight(mtp[k].float())
    sd.update(mtp)
    save_file(sd, path / "model.safetensors")
    conf = json.loads((path / "config.json").read_text())
    conf["mtp_num_hidden_layers"] = 1
    if "quantization_config" in conf:
        q = conf["quantization_config"]
        q["modules_to_not_convert"] = sorted({*q["modules_to_not_convert"], "mtp.fc", "mtp.norm",
                                              "mtp.pre_fc_norm_embedding", "mtp.pre_fc_norm_hidden",
                                              "mtp.layers.0.input_layernorm", "mtp.layers.0.post_attention_layernorm",
                                              "mtp.layers.0.self_attn.q_norm", "mtp.layers.0.self_attn.k_norm"})
    (path / "config.json").write_text(json.dumps(conf, indent=1))


def make_tiny_model(path: Path, *, vocab: int = 512, layers: tuple[str, ...] = LAYERS, seed: int = 0,
                    tie: bool = False, scale: float = 0.05, prefix: str | None = None,
                    tokenizer_from: Path | None = None, mtp: bool = False, gpu: bool = False,
                    bf16: bool = False, attn: tuple[int, int, int] | None = None) -> Path:
    """Save a random Qwen3_5ForCausalLM (both layer kinds, GQA in attention and in GDN) as safetensors.

    Every parameter gets noise on top of HF's init, so the (1 + w) norms, the gated norm weight and dt_bias are
    exercised away from their identity values. `prefix` renames `model.` to e.g. `model.language_model.` (the
    Qwen3_5ForConditionalGeneration layout); `tokenizer_from` copies tokenizer files next to the weights; `mtp` adds
    an MTP layer (`add_mtp_layer`); `gpu`: dims our CUDA kernels take (`GPU_DIMS`); `bf16`: saved in BF16 (`to_bf16`),
    so transformers in fp32 runs exactly the weights a BF16 runner does; `attn`: (query heads, KV heads, head dim) of
    the attention layers instead (`QWEN38_ATTN`, for batch invariance's attention kernel)."""
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    dims = dict(GPU_DIMS if gpu else CPU_DIMS)
    if attn:
        dims.update(num_attention_heads=attn[0], num_key_value_heads=attn[1], head_dim=attn[2])
    cfg = Qwen3_5TextConfig(
        vocab_size=vocab, num_hidden_layers=len(layers), layer_types=list(layers), linear_num_key_heads=2,
        linear_num_value_heads=4, linear_conv_kernel_dim=4, max_position_embeddings=4096,
        rms_norm_eps=1e-6, tie_word_embeddings=tie, eos_token_id=vocab - 1,
        rope_parameters={"rope_type": "default", "rope_theta": 10_000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True}, **dims)
    torch.manual_seed(seed)
    model = Qwen3_5ForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("A_log"):
                continue
            p.add_(torch.randn_like(p) * (scale if p.dim() > 1 else 0.3))
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    if prefix:
        from safetensors.torch import load_file, save_file

        f = path / "model.safetensors"
        sd = load_file(f)
        save_file({(prefix + k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}, f)
    if tokenizer_from is not None:
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
            if (tokenizer_from / name).exists():
                shutil.copy(tokenizer_from / name, path / name)
    if mtp:
        add_mtp_layer(path, model, seed, scale)
    if bf16:
        to_bf16(path)
    return path


def to_bf16(path: Path) -> None:
    """Save a checkpoint's floating-point tensors in BF16."""
    from safetensors.torch import load_file, save_file

    f = path / "model.safetensors"
    save_file({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in load_file(f).items()}, f)


VL_TOKENS = {"vision_start_token_id": 497, "vision_end_token_id": 498, "image_token_id": 499, "video_token_id": 496}
VL_VISION = {"depth": 2, "hidden_size": 64, "num_heads": 4, "intermediate_size": 96, "patch_size": 16,
             "temporal_patch_size": 2, "spatial_merge_size": 2, "out_hidden_size": 64, "num_position_embeddings": 64,
             "in_channels": 3, "hidden_act": "gelu_pytorch_tanh", "deepstack_visual_indexes": []}


def make_tiny_vl_model(path: Path, *, vocab: int = 512, layers: tuple[str, ...] = LAYERS, seed: int = 0,
                       scale: float = 0.05, mtp: bool = False, gpu: bool = False):
    """A random Qwen3_5ForConditionalGeneration (the tiny text model of `make_tiny_model` plus a 2-block vision tower
    with the real patch geometry), saved under the checkpoint's names (`model.language_model.*`, `model.visual.*`,
    `lm_head`) with a preprocessor_config.json. Returns the HF model (fp32) as the reference. `gpu`: dims our CUDA
    kernels take (hidden 128, attention heads of 64, GDN heads of 128)."""
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    dims = GPU_DIMS if gpu else CPU_DIMS
    text = dict(vocab_size=vocab, num_hidden_layers=len(layers), layer_types=list(layers), linear_num_key_heads=2,
                linear_num_value_heads=4, linear_conv_kernel_dim=4, max_position_embeddings=4096, rms_norm_eps=1e-6,
                tie_word_embeddings=False, eos_token_id=vocab - 1,
                rope_parameters={"rope_type": "default", "rope_theta": 10_000.0, "partial_rotary_factor": 0.25,
                                 "mrope_section": [2, 1, 1], "mrope_interleaved": True}, **dims)
    vision = {k: v for k, v in VL_VISION.items() if k != "deepstack_visual_indexes"}
    vision["out_hidden_size"] = dims["hidden_size"]
    cfg = Qwen3_5Config(text_config=text, vision_config=vision, tie_word_embeddings=False, **VL_TOKENS)
    torch.manual_seed(seed)
    model = Qwen3_5ForConditionalGeneration(cfg).float().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not name.endswith("A_log"):
                p.add_(torch.randn_like(p) * (scale if p.dim() > 1 else 0.3))
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    conf = json.loads((path / "config.json").read_text())
    conf["vision_config"]["deepstack_visual_indexes"] = []
    conf["vision_config"]["out_hidden_size"] = dims["hidden_size"]
    (path / "config.json").write_text(json.dumps(conf, indent=1))
    (path / "preprocessor_config.json").write_text(json.dumps({
        "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2, "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5], "size": {"shortest_edge": 1024, "longest_edge": 1 << 20}}))
    if mtp:
        text_model = model.model.language_model
        add_mtp_layer(path, type("M", (), {"config": text_model.config})(), seed, scale)
        conf = json.loads((path / "config.json").read_text())
        conf["text_config"]["mtp_num_hidden_layers"] = conf.pop("mtp_num_hidden_layers")
        (path / "config.json").write_text(json.dumps(conf, indent=1))
    return model


FP8_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj",
                   "up_proj", "down_proj")


def make_tiny_fp8_model(path: Path, *, seed: int = 0, vocab: int = 512, layers: tuple[str, ...] = LAYERS,
                        scale: float = 0.05, mtp: bool = False, gpu: bool = False,
                        attn: tuple[int, int, int] | None = None) -> tuple[Path, Path]:
    """A random Qwen3.5 in Qwen's block-FP8 format (e4m3 projections with `weight_scale_inv` per 128 × 128 block,
    `quantization_config` with `modules_to_not_convert`), every projection dim a multiple of 128: hidden 128,
    attention 2 heads of 64 (gated q 256, k and v 128), GDN 2 key heads of 64 and 4 value heads of 32 (in_proj_qkvz
    512, in_proj_ba 8 and BF16, as in the real checkpoints), MLP 256. Returns (the FP8 checkpoint, the same model with
    the dequantized weights in fp32): transformers runs the latter as the reference. `gpu`: every head of 128, in
    attention (gated q 512, k and v 256) and in GDN (in_proj_qkvz 1,536), so our CUDA kernels serve every layer (the GDN
    kernels, FlashInfer's GDN prefill, the quantizing gate) and the runner captures CUDA graphs; `mtp`: the FP8
    checkpoint gets a block-FP8 MTP layer (`add_mtp_layer`), as the real ones have. `attn`: (query heads, KV heads,
    head dim) of the attention layers instead (`QWEN38_ATTN`: Qwen3.8 27B's head dim and GQA group, for batch
    invariance's attention kernel)."""
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    d, dk, dv = (128, 128, 128) if gpu else (64, 64, 32)
    h, hkv, d = attn or (2, 2, d)
    cfg = Qwen3_5TextConfig(
        vocab_size=vocab, hidden_size=128, intermediate_size=256, num_hidden_layers=len(layers),
        layer_types=list(layers), num_attention_heads=h, num_key_value_heads=hkv, head_dim=d,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=dk, linear_value_head_dim=dv,
        linear_conv_kernel_dim=4, max_position_embeddings=4096, rms_norm_eps=1e-6, tie_word_embeddings=False,
        eos_token_id=vocab - 1,
        rope_parameters={"rope_type": "default", "rope_theta": 10_000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True})
    torch.manual_seed(seed)
    model = Qwen3_5ForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not name.endswith("A_log"):
                p.add_(torch.randn_like(p) * (scale if p.dim() > 1 else 0.3))
    deq_dir, fp8_dir = path / "dequantized", path / "fp8"
    model.save_pretrained(deq_dir, safe_serialization=True)
    model.save_pretrained(fp8_dir, safe_serialization=True)
    quantize_checkpoint(fp8_dir, deq_dir, [f"model.layers.{i}.linear_attn.in_proj_ba"
                                           for i, t in enumerate(layers) if t == "linear_attention"])
    if mtp:
        add_mtp_layer(fp8_dir, model, seed, scale, fp8=True)
    return fp8_dir, deq_dir


QWEN38_ATTN = (6, 1, 256)  # query heads, KV heads, head dim: Qwen3.8 27B's head dim and GQA group (its 24 over 4)


def quantize_checkpoint(fp8_dir: Path, deq_dir: Path | None, extra_keep: list[str]) -> None:
    """Rewrite `fp8_dir`'s safetensors in Qwen's block-FP8 format: every projection of FP8_PROJECTIONS as e4m3 with a
    `weight_scale_inv` per 128 x 128 block, the rest BF16 (vectors as they are), and config.json's
    `quantization_config` listing the rest (`modules_to_not_convert`, with `extra_keep`: names the real checkpoints
    give their BF16 modules); `deq_dir` (or None) gets the dequantized weights."""
    from safetensors.torch import load_file, save_file

    from llm_mini_infer.kernels import fp8

    sd = load_file(fp8_dir / "model.safetensors")
    quantized, deq = {}, {}
    for k, v in sd.items():
        if k.endswith(".weight") and k.split(".")[-2] in FP8_PROJECTIONS:
            q, sc = fp8.quantize_weight(v.float())
            quantized[k], quantized[k + "_scale_inv"] = q, sc
            deq[k] = fp8.dequant_weight(q, sc)
        else:
            quantized[k] = deq[k] = v.to(torch.bfloat16) if v.dim() > 1 else v
    if deq_dir is not None:
        save_file(deq, deq_dir / "model.safetensors")
    save_file(quantized, fp8_dir / "model.safetensors")
    keep = [k.removesuffix(".weight") for k in sd if k.split(".")[-2] not in FP8_PROJECTIONS] + extra_keep
    conf = json.loads((fp8_dir / "config.json").read_text())
    conf["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
                                   "weight_block_size": [128, 128], "modules_to_not_convert": sorted(keep)}
    (fp8_dir / "config.json").write_text(json.dumps(conf, indent=1))


def make_tiny_vl_fp8_model(path: Path, *, seed: int = 0, scale: float = 0.05, mtp: bool = False,
                           attn: tuple[int, int, int] = QWEN38_ATTN) -> Path:
    """`make_tiny_vl_model`'s multimodal Qwen (`gpu` dims, attention `attn`) in Qwen's block-FP8 format: the text
    model's projections FP8, its tower BF16 (`quantize_checkpoint`), as a real FP8 checkpoint's; with `mtp` its
    block-FP8 MTP layer. Returns the checkpoint's folder."""
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    h, hkv, d = attn
    dims = dict(GPU_DIMS, num_attention_heads=h, num_key_value_heads=hkv, head_dim=d)
    text = dict(vocab_size=512, num_hidden_layers=len(LAYERS), layer_types=list(LAYERS), linear_num_key_heads=2,
                linear_num_value_heads=4, linear_conv_kernel_dim=4, max_position_embeddings=4096, rms_norm_eps=1e-6,
                tie_word_embeddings=False, eos_token_id=511,
                rope_parameters={"rope_type": "default", "rope_theta": 10_000.0, "partial_rotary_factor": 0.25,
                                 "mrope_section": [2, 1, 1], "mrope_interleaved": True}, **dims)
    vision = {k: v for k, v in VL_VISION.items() if k != "deepstack_visual_indexes"}
    vision["out_hidden_size"] = dims["hidden_size"]
    cfg = Qwen3_5Config(text_config=text, vision_config=vision, tie_word_embeddings=False, **VL_TOKENS)
    torch.manual_seed(seed)
    model = Qwen3_5ForConditionalGeneration(cfg).float().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not name.endswith("A_log"):
                p.add_(torch.randn_like(p) * (scale if p.dim() > 1 else 0.3))
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    conf = json.loads((path / "config.json").read_text())
    conf["vision_config"]["deepstack_visual_indexes"] = []
    conf["vision_config"]["out_hidden_size"] = dims["hidden_size"]
    (path / "config.json").write_text(json.dumps(conf, indent=1))
    (path / "preprocessor_config.json").write_text(json.dumps({
        "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2, "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5], "size": {"shortest_edge": 1024, "longest_edge": 1 << 20}}))
    quantize_checkpoint(path, None, [f"model.language_model.layers.{i}.linear_attn.in_proj_ba"
                                     for i, t in enumerate(LAYERS) if t == "linear_attention"])
    if mtp:
        text_model = model.model.language_model
        add_mtp_layer(path, type("M", (), {"config": text_model.config})(), seed, scale, fp8=True)
        conf = json.loads((path / "config.json").read_text())
        conf["text_config"]["mtp_num_hidden_layers"] = conf.pop("mtp_num_hidden_layers")
        (path / "config.json").write_text(json.dumps(conf, indent=1))
    return path


def hf_logits(model_dir: Path, ids: list[int]) -> torch.Tensor:
    """Reference logits [T, V] (fp32) from transformers' implementation, one forward over the whole sequence."""
    from transformers import Qwen3_5ForCausalLM

    model = Qwen3_5ForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    with torch.no_grad():
        return model(torch.tensor([ids])).logits[0].float()


@dataclass
class Row:
    slot: int
    tokens: list[int] | None  # None: device input (decode)
    pos0: int  # for device rows: the position the runner reads from its seq_len (used for page coverage)
    sample: bool = True
    next_tok: int = -1  # MTP: the token after a non-final chunk
    precise: bool = False  # PRECISE: BF16 activations (the precise prefix)


@dataclass
class Step:
    rows: list[Row] = field(default_factory=list)
    free: list[int] = field(default_factory=list)
    admit: list[tuple[int, int, tuple[float, float, int, float, int]]] = field(default_factory=list)
    seq_len: dict[int, int] = field(default_factory=dict)  # slot -> ADMIT seq_len (a cached prefix)
    set_pages: dict[int, list[int]] = field(default_factory=dict)  # slot -> its whole page list (shared pages)
    kv_copy: list[tuple[int, int]] = field(default_factory=list)
    state_copy: list[tuple[int, int]] = field(default_factory=list)
    ckpt: list[tuple[int, int, int]] = field(default_factory=list)  # (row in SCHED order, offset, dst state slot)
    fold: list[tuple[int, int]] = field(default_factory=list)  # (slot, records) before FREE (MTP)
    stop: dict[int, list[int]] = field(default_factory=dict)  # slot -> its commit limit's stop tokens (MTP)


class Driver:
    """Builds plans by hand for a CudaRunner, each just before it is launched (PlanBuilder reuses its buffers two
    plans later), runs them in the engine's call order, and collects the logits of every computed position
    through the runner's logits hook, keyed by (slot, position)."""

    def __init__(self, runner: CudaRunner, pools: PoolConfig, capture: bool = True):
        self.r = runner
        self.pools = pools
        self.b = PlanBuilder(pools.max_num_seqs, pools.max_num_batched_tokens, pools.page_size, pools.max_model_len,
                             pools.mtp_k)
        self.step_id = 0
        self.next_page = 1
        self.pages: dict[int, list[int]] = {}
        self.free_pages: list[int] = []
        self.logits: dict[tuple[int, int], torch.Tensor] = {}
        self._slots: list[int] = []
        runner.logits_hook = self._hook if capture else None

    def _hook(self, chunk: LogitsChunk) -> None:
        lg = chunk.logits.detach().to("cpu", torch.float32)
        for i, (row, pos) in enumerate(zip(chunk.rows.tolist(), chunk.positions.tolist())):
            self.logits[(self._slots[row], pos)] = lg[i]

    def _alloc(self) -> int:
        if self.free_pages:
            return self.free_pages.pop()
        self.next_page += 1
        if self.next_page > self.pools.num_pages:
            raise RuntimeError("the test pool is out of pages")
        return self.next_page - 1

    def _cover(self, slot: int, upto: int) -> None:
        have = self.pages.setdefault(slot, [])
        new = [self._alloc() for _ in range(max(0, -(-upto // self.pools.page_size) - len(have)))]
        if new:
            self.b.add_pages(slot, len(have), new)
            have.extend(new)

    def plan(self, step: Step) -> StepPlan:
        b = self.b
        b.begin()
        for s, n in step.fold:
            b.add_fold(s, n)
        for s in step.free:
            b.add_free(s)
            self.free_pages.extend(self.pages.pop(s, []))
        for slot, ss, (temp, top_p, top_k, min_p, seed) in step.admit:
            b.add_admit(slot, ss, step.seq_len.get(slot, 0), temp, top_p, top_k, min_p, seed,
                        max_len=self.pools.max_model_len, stop=step.stop.get(slot, ()))
        for slot, pages in step.set_pages.items():
            b.add_pages(slot, 0, pages)
            self.pages[slot] = list(pages)
        for src, dst in step.kv_copy:
            b.add_kv_copy(src, dst)
        for src, dst in step.state_copy:
            b.add_state_copy(src, dst)
        dev = [r for r in step.rows if r.tokens is None]
        self._slots = [r.slot for r in dev]
        k = self.pools.mtp_k
        for r in dev:
            self._cover(r.slot, r.pos0 + (1 if not k else 3 * k + 1))
        if dev:
            b.add_device_rows(np.asarray([r.slot for r in dev], np.int32))
        for r in step.rows:
            if r.tokens is None:
                continue
            self._cover(r.slot, r.pos0 + len(r.tokens) + (2 * k + 1 if k and r.sample else 0))
            b.add_explicit_row(r.slot, np.asarray(r.tokens, np.int32), r.pos0,
                               (STATE_ZERO if r.pos0 == 0 else 0) | (SAMPLE if r.sample else 0) |
                               (PRECISE if r.precise else 0),
                               -1 if r.sample or not k else int(r.next_tok))
            self._slots.append(r.slot)
        for row, off, dst in step.ckpt:
            b.add_ckpt(row, off, dst)
        self.step_id += 1
        return b.finish(self.step_id)

    def alloc_pages(self, n: int) -> list[int]:
        return [self._alloc() for _ in range(n)]

    def run(self, steps: list[Step]) -> list[StepResult]:
        """Plan and launch each step in the engine's order (launch N+1, wait N, sample N+1)."""
        out, prev = [], None
        for s in steps:
            h = self.r.launch_forward(self.plan(s))
            if prev is not None:
                out.append(self.r.wait(prev))
            self.r.launch_sample(h)
            prev = h
        if prev is not None:
            out.append(self.r.wait(prev))
        return out


def pools(max_num_seqs: int = 4, budget: int = 256, page_size: int = 4, num_pages: int = 512, state_slots: int = 0,
          max_model_len: int = 1024, mtp_k: int = 0) -> PoolConfig:
    return PoolConfig(page_size=page_size, num_pages=num_pages, num_state_slots=state_slots or max_num_seqs + 1,
                      max_num_seqs=max_num_seqs, max_num_batched_tokens=budget, max_model_len=max_model_len,
                      eos_token_ids=(), mtp_k=mtp_k)


def cpu_runner(model_dir: Path, pc: PoolConfig, **kw) -> CudaRunner:
    r = CudaRunner(model_dir, device="cpu", dtype=torch.float32, max_num_seqs=pc.max_num_seqs,
                   max_num_batched_tokens=pc.max_num_batched_tokens, max_model_len=pc.max_model_len, check=True,
                   mtp_k=pc.mtp_k, **kw)
    r.configure(pc)
    return r


def cuda_runner(model_dir: Path, pc: PoolConfig, graphs: bool = True, check: bool = True, **kw) -> CudaRunner:
    """`cpu_runner` on CUDA: BF16, our kernels, FlashInfer's attention and GDN prefill (the `gpu` dims) and, with
    `graphs`, CUDA graphs; the pools `pc` sets, no memory profile. `check` synchronizes the device at every
    launch_sample (the host mirror)."""
    r = CudaRunner(model_dir, device="cuda", max_num_seqs=pc.max_num_seqs,
                   max_num_batched_tokens=pc.max_num_batched_tokens, max_model_len=pc.max_model_len, memory_gb=1.0,
                   check=check, graphs=graphs, mtp_k=pc.mtp_k, **kw)
    r.configure(pc)
    return r


GREEDY = (0.0, 1.0, 0, 0.0, 1)
