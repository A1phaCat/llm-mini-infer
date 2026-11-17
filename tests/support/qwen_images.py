"""Qwen's image tests: the tiny multimodal model, images, prompts and the engine (tests/runner/test_cuda_images.py)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from llm_mini_infer.engine.core import EngineCore
from llm_mini_infer.engine.messages import AddRequest, ImageRef
from llm_mini_infer.runner.base import PoolConfig
from llm_mini_infer.runner.cuda.runner import CudaRunner, VisionOptions
from llm_mini_infer.sched.scheduler import Scheduler
from llm_mini_infer.vision import arena as px_arena
from llm_mini_infer.vision.family import KEY_BASE
from llm_mini_infer.vision.qwen import QwenVision

from .tiny_qwen import VL_TOKENS, make_tiny_vl_model

START, END, PAD = VL_TOKENS["vision_start_token_id"], VL_TOKENS["vision_end_token_id"], VL_TOKENS["image_token_id"]


@pytest.fixture(scope="module")
def vl(tmp_path_factory):
    path = tmp_path_factory.mktemp("vl")
    model = make_tiny_vl_model(path, seed=2, mtp=True)
    return path, model


def text(n: int, salt: int = 0) -> list[int]:
    return [(i * 53 + 7 + salt * 131) % 490 for i in range(n)]


class Img:
    def __init__(self, gh: int, gw: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.gh, self.gw = gh, gw
        self.px = torch.randint(0, 256, (3, 16 * gh, 16 * gw), generator=g, dtype=torch.uint8)
        self.tokens = gh * gw // 4
        self.key = KEY_BASE | seed


def prompt_of(parts: list) -> tuple[list[int], list[tuple[int, Img]]]:
    """Text lists and images in order -> token IDs (each image wrapped in its start and end tokens) and spans."""
    ids: list[int] = []
    spans = []
    for x in parts:
        if isinstance(x, Img):
            ids.append(START)
            spans.append((len(ids), x))
            ids += [PAD] * x.tokens
            ids.append(END)
        else:
            ids += x
    return ids, spans


class Engine:
    """A CPU runner with images (`runner_kw`: more of its options), the scheduler and the engine loop; the logits hook
    keyed by (rid, position)."""

    def __init__(self, path, tmp, *, budget: int = 128, k: int = 0, caching: bool = False, enc_rows: int = 256,
                 enc_budget: int = 64, max_num_seqs: int = 2, device: str = "cpu", **runner_kw):
        self.arena_path = str(tmp / f"arena-{budget}-{k}-{caching}-{device}")
        px_arena.remove(self.arena_path)
        px_arena.create(self.arena_path, 1 << 22)
        self.arena = px_arena.map_writable(self.arena_path)
        self.next_px = 0
        cuda = device == "cuda"
        self.runner = CudaRunner(path, device=device, dtype=torch.bfloat16 if cuda else torch.float32,
                                 max_num_seqs=max_num_seqs, max_num_batched_tokens=budget, max_model_len=512,
                                 check=True, mtp_k=k, memory_gb=1.0 if cuda else None,
                                 token_buckets=[8, 16] if cuda else None,
                                 vision=VisionOptions(encoder_rows=enc_rows, encoder_budget=enc_budget,
                                                      arena=self.arena_path), **runner_kw)
        nss = max_num_seqs + 1 + 6
        self.runner.configure(PoolConfig(page_size=4, num_pages=300, num_state_slots=nss, max_num_seqs=max_num_seqs,
                                         max_num_batched_tokens=budget, max_model_len=512, eos_token_ids=(),
                                         mtp_k=k))
        caps = self.runner.capabilities()
        fam = QwenVision.from_model_dir(path, json.loads((path / "config.json").read_text()))
        self.sched = Scheduler(max_num_seqs=max_num_seqs, max_num_batched_tokens=budget, max_model_len=512,
                               page_size=4, num_pages=300, num_state_slots=nss, eos_token_ids=(), mtp_k=k,
                               prefix_caching=caching, decode_checkpoint_interval=64, encoder_rows=caps.encoder_rows,
                               encoder_budget=caps.encoder_budget, positions=fam.positions)
        self.core = EngineCore(self.sched, self.runner)
        self.logits: dict[tuple[int, int], torch.Tensor] = {}
        self._slot_rid: dict[int, dict[int, int]] = {}
        orig = self.runner.launch_forward

        def record(plan, _internal=False, _capture=None):
            if not _internal:
                self._slot_rid[plan.step_id] = {int(s): int(self.sched.fe_rid[s]) for s in plan.sched_slot}
                self._rows = plan.sched_slot.tolist()
            return orig(plan, _internal=_internal, _capture=_capture)

        self.runner.launch_forward = record

        def hook(c) -> None:
            slots = list(self._rows)
            for i, (row, pos) in enumerate(zip(c.rows.tolist(), c.positions.tolist())):
                rid = self._slot_rid[c.step_id][slots[row]] if c.step_id in self._slot_rid else -1
                self.logits[(rid, pos)] = c.logits[i].detach().float().cpu().clone()

        self.runner.logits_hook = hook
        self.out: dict[int, list[int]] = {}

    def add(self, rid: int, ids: list[int], spans, max_tokens: int = 4) -> None:
        refs = []
        for off, im in spans:
            n = im.px.numel()
            np.frombuffer(self.arena.buf, np.uint8, n, self.next_px)[:] = im.px.numpy().reshape(-1)
            refs.append(ImageRef(key=im.key, offset=off, tokens=im.tokens, grid_t=1, grid_h=im.gh, grid_w=im.gw,
                                 height=16 * im.gh, width=16 * im.gw, pixels=self.next_px))
            self.next_px += -(-n // 4096) * 4096
        self.out[rid] = []
        self.core.handle(AddRequest(rid=rid, frontend=0, prompt=np.asarray(ids, np.int32), max_tokens=max_tokens,
                                    temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=100 + rid,
                                    stop_token_ids=[], ignore_eos=True, images=refs))

    def run(self) -> None:
        for _ in range(3000):
            report = self.core.step()
            if report is not None:
                for fo in report.outputs.values():
                    off = 0
                    for rid, n in zip(fo.rids.tolist(), fo.n_new.tolist()):
                        self.out[rid].extend(fo.tokens[off:off + n].tolist())
                        off += n
            if not self.core.busy:
                self.sched.check_accounting()
                return
        raise AssertionError("engine not idle")

    def prompt_logits(self, rid: int, n: int) -> torch.Tensor:
        return torch.stack([self.logits[(rid, p)] for p in range(n)])
