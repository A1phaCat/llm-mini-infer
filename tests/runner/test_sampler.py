"""Counter-based Gumbel-max sampling (runner/cuda/sampler.py): exact hashing, batch independence, filters, ties, the
sampled distribution, the noise's bound and non-finite rows; padded id maps and masks tested through the ids (the
grammar-sparse head samples the union of the step's masks exactly as the dense logits; architecture.md,
"Grammar-aware heads"); the sampled frequencies' goodness of fit. tests/kernels/test_kernels.py checks the fused CUDA
sampler against it."""

from __future__ import annotations

import numpy as np
import torch

from llm_mini_infer.runner.cuda.sampler import MASK_EMPTY, NONFINITE, PHI, plan_sampling, row_keys, sample, splitmix64
from llm_mini_infer.runner.fake import MASK64
from llm_mini_infer.runner.fake import splitmix64 as splitmix64_ref

from ..support.sampler_cases import (
    FIT_CASES,
    FIT_DRAWS,
    FIT_SEED,
    assert_fits,
    fit_row,
    largest_noise_rows,
    target_distribution,
)


def run(logits: torch.Tensor, temp, top_p, top_k, min_p, seed, pos) -> list[int]:
    n = logits.shape[0]
    arr = lambda x, dt: np.broadcast_to(np.asarray(x, dt), (n,)).copy()  # noqa: E731
    t, p, k, m = arr(temp, np.float32), arr(top_p, np.float32), arr(top_k, np.int64), arr(min_p, np.float32)
    plan = plan_sampling(t, p, k, m, logits.shape[1])
    for g in plan.groups:
        g.idx = torch.as_tensor(g.rows)
    keys = row_keys(torch.as_tensor(arr(seed, np.int64)), torch.as_tensor(arr(pos, np.int64)))
    out, status = sample(logits, plan, torch.as_tensor(t), torch.as_tensor(p), torch.as_tensor(k), torch.as_tensor(m),
                         keys, torch.arange(logits.shape[1]))
    assert not status.any()
    return out.tolist()


def test_splitmix64_is_the_exact_64_bit_hash() -> None:
    rng = np.random.default_rng(0)
    xs = rng.integers(-(1 << 63), (1 << 63) - 1, size=1000, dtype=np.int64)
    got = splitmix64(torch.as_tensor(xs)).numpy().view(np.uint64)
    want = np.array([splitmix64_ref(int(x) & MASK64) for x in xs], dtype=np.uint64)
    assert np.array_equal(got, want)
    assert PHI & MASK64 == 0x9E3779B97F4A7C15


def test_tokens_depend_on_seed_and_position_only() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 1000)
    seeds, positions = [3, 3, 9, 12345678901234], [10, 11, 10, 7]
    alone = [run(logits, 1.0, 1.0, 0, 0.0, s, p)[0] for s, p in zip(seeds, positions)]
    batched = run(logits.expand(4, -1).contiguous(), 1.0, 1.0, 0, 0.0, seeds, positions)
    assert batched == alone
    # Inside a batch of other rows with other filters, in another order: still the same tokens.
    other = torch.randn(3, 1000)
    mixed = torch.cat([other, logits.expand(4, -1)])[[3, 0, 4, 1, 5, 2, 6]]  # L, O0, L, O1, L, O2, L
    got = run(mixed, [1.0, 0.0, 1.0, 0.7, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0], [0, 0, 0, 5, 0, 3, 0],
              0.0, [3, 1, 3, 2, 9, 4, 12345678901234], [10, 1, 11, 2, 10, 3, 7])
    assert [got[0], got[2], got[4], got[6]] == alone
    assert len(set(alone)) > 1  # different (seed, position) draw differently


def test_greedy_and_top_k_1_are_argmax() -> None:
    torch.manual_seed(1)
    logits = torch.randn(16, 300)
    best = logits.argmax(-1).tolist()
    assert run(logits, 0.0, 1.0, 0, 0.0, range(16), 5) == best
    assert run(logits, 1.0, 1.0, 1, 0.0, range(16), 5) == best
    assert run(logits, 1e-6, 1.0, 0, 0.0, range(16), 5) == best  # below the sampler's 1e-5: greedy
    for p in (1e-9, 1e-12, 1e-45, 1e-46):  # any top_p below 1 / the vocabulary keeps the top token (1e-46: 0 in fp32)
        assert run(logits, 1.0, p, 0, 0.0, range(16), 5) == best, p


def test_distribution_and_filters() -> None:
    logits = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -3.0])
    probs = logits.softmax(-1)
    n = 40_000
    many = logits.expand(n, -1).contiguous()

    def freq(**kw) -> np.ndarray:
        args = {"temp": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, **kw}
        toks = run(many, args["temp"], args["top_p"], args["top_k"], args["min_p"], 42, np.arange(n))
        return np.bincount(toks, minlength=8) / n

    f = freq()
    sigma = np.sqrt(probs.numpy() * (1 - probs.numpy()) / n)
    assert (np.abs(f - probs.numpy()) < 5 * sigma + 1e-4).all()
    f = freq(top_k=3)
    p3 = probs[:3] / probs[:3].sum()
    assert f[3:].sum() == 0 and np.abs(f[:3] - p3.numpy()).max() < 0.01
    # top-p 0.6: the nucleus is the smallest prefix reaching 0.6 (here the first two tokens)
    assert np.cumsum(probs.numpy())[0] < 0.6 <= np.cumsum(probs.numpy())[1]
    f = freq(top_p=0.6)
    assert f[2:].sum() == 0 and abs(f[0] - probs[0] / probs[:2].sum()) < 0.01
    # min-p 0.3: tokens with p >= 0.3 * p_max survive
    keep = (probs >= 0.3 * probs.max()).numpy()
    f = freq(min_p=0.3)
    assert f[~keep].sum() == 0 and (f[keep] > 0).all()
    assert freq(temp=0.5)[0] > freq()[0] + 0.05  # temperature sharpens


def test_ties_are_ordered_by_id() -> None:
    """Five tokens share the top logit (BF16 logits tie often): top-k 3 keeps the three lowest ids among them, and
    top-p over equal masses keeps the lowest ids too."""
    logits = torch.full((1, 64), -5.0)
    tied = [40, 3, 17, 60, 9]
    logits[0, tied] = 2.0
    n = 4000
    many = logits.expand(n, -1).contiguous()
    toks = run(many, 1.0, 1.0, 3, 0.0, 7, np.arange(n))
    assert set(toks) == {3, 9, 17}
    # top-p 0.5 of five equal masses (plus negligible others): the first three in id order (excl. mass 0, 0.2, 0.4)
    toks = run(many, 1.0, 0.5, 0, 0.0, 7, np.arange(n))
    assert set(toks) == {3, 9, 17}


def test_nonfinite_rows_are_flagged() -> None:
    logits = torch.randn(4, 50)
    logits[1, 7] = float("nan")
    logits[3, 0] = float("inf")
    t = np.array([1.0, 1.0, 0.0, 0.0], np.float32)
    plan = plan_sampling(t, np.ones(4, np.float32), np.zeros(4, np.int64), np.zeros(4, np.float32), 50)
    for g in plan.groups:
        g.idx = torch.as_tensor(g.rows)
    keys = row_keys(torch.arange(4), torch.arange(4))
    out, status = sample(logits, plan, torch.as_tensor(t), torch.ones(4), torch.zeros(4, dtype=torch.int64),
                         torch.zeros(4), keys, torch.arange(50))
    assert status.tolist() == [0, 1, 0, 1] and out[1] == 0 and out[3] == 0


def test_a_reduced_vocabulary_samples_as_the_full_one_with_the_rest_masked() -> None:
    """The MTP drafter's head: logits over a subset of the vocabulary with its token ids (ascending) give the tokens
    that sampling the whole vocabulary gives with every other token's logit at -1e30, for every filter: the noise is
    the true id's and ties still go to the lowest id."""
    torch.manual_seed(3)
    v, n = 3000, 12
    ids = torch.as_tensor(np.sort(np.random.default_rng(4).choice(v, 700, replace=False)))
    full = torch.full((n, v), -1e30)
    sub = torch.randn(n, len(ids)) * torch.linspace(1, 6, n)[:, None]
    sub[:, 5] = sub[:, 9]  # a tie
    full[:, ids] = sub
    combos = [(0.0, 1.0, 0, 0.0), (1.0, 1.0, 0, 0.0), (0.7, 0.8, 20, 0.0), (1.0, 0.9, 0, 0.0), (1.2, 1.0, 50, 0.05)]
    for temp, top_p, top_k, min_p in combos:
        seed, pos = np.arange(n) + 11, np.arange(n) * 7 + 3
        arr = lambda x, dt: np.full(n, x, dt)  # noqa: E731
        t, p, k, m = arr(temp, np.float32), arr(top_p, np.float32), arr(top_k, np.int64), arr(min_p, np.float32)
        keys = row_keys(torch.as_tensor(seed), torch.as_tensor(pos))
        params = (torch.as_tensor(t), torch.as_tensor(p), torch.as_tensor(k), torch.as_tensor(m), keys)
        got = []
        for logits, vocab_ids in ((sub, ids), (full, torch.arange(v))):
            plan = plan_sampling(t, p, k, m, logits.shape[1])
            for g in plan.groups:
                g.idx = torch.as_tensor(g.rows)
            out, status = sample(logits, plan, *params, vocab_ids)
            assert not status.any()
            got.append(out.tolist())
        assert got[0] == got[1], (temp, top_p, top_k, min_p)
        assert set(got[0]) <= set(ids.tolist())


def test_an_excluded_token_is_never_sampled_whatever_its_noise() -> None:
    """One token in 2^24 draws the top uniform, which rounds to 1 in fp32 (noise +inf) and is clamped to the largest
    float below 1, in the kernel as here: its noise is 16.64, the largest. Seed 5738 at position 1 draws it on token
    3980 of a 4096 vocabulary. On a token already excluded (logit -inf after min-p, top-k, top-p or a grammar mask) it
    stays excluded; on a candidate far below the others it is too small to win, where +inf would."""
    from llm_mini_infer.runner.cuda.sampler import U_MAX, gumbel, unpack_masks

    v, seed, pos, tok = 4096, 5738, 1, 3980
    keys = row_keys(torch.as_tensor([seed]), torch.as_tensor([pos]))
    noise = gumbel(keys, torch.arange(v))[0]
    top = -torch.log(-torch.log(torch.tensor(U_MAX)))
    assert torch.isfinite(noise).all() and 16.63 < top < 16.64
    assert (noise == noise.max()).nonzero()[:, 0].tolist() == [tok] and noise[tok] == top
    torch.manual_seed(9)
    logits = torch.randn(1, v)
    logits[0, tok] = logits.max() + 0.1  # the largest logit, excluded below
    allowed = torch.ones(1, v, dtype=torch.bool)
    allowed[0, tok] = False
    words = torch.as_tensor(np.packbits(allowed.numpy(), axis=1, bitorder="little").view(np.int32))
    assert torch.equal(unpack_masks(words, v), allowed)
    for temp, top_p, top_k, min_p in [(1.0, 1.0, 0, 0.0), (1.0, 0.9, 0, 0.0), (1.0, 1.0, 50, 0.0), (1.0, 1.0, 0, 0.1)]:
        t, p, k, m = (np.array([x], dt) for x, dt in ((temp, np.float32), (top_p, np.float32), (top_k, np.int64),
                                                        (min_p, np.float32)))
        plan = plan_sampling(t, p, k, m, v)
        for g in plan.groups:
            g.idx = torch.as_tensor(g.rows)
        out, status = sample(logits, plan, torch.as_tensor(t), torch.as_tensor(p), torch.as_tensor(k),
                             torch.as_tensor(m), keys, torch.arange(v), mask=words,
                             mask_row=torch.zeros(1, dtype=torch.int64))
        assert not status.any() and out.item() != tok, (temp, top_p, top_k, min_p)
    # Unmasked, with the token's logit far below the others': outside a min-p cut and a top-p nucleus, excluded the
    # same way; with neither, a candidate whose noise falls short of the gap.
    low = logits.clone()
    low[0, tok] = logits.min() - 50
    assert run(low, 1.0, 1.0, 0, 0.1, seed, pos)[0] != tok
    assert run(low, 1.0, 0.5, 0, 0.0, seed, pos)[0] != tok
    assert run(low, 1.0, 1.0, 0, 0.0, seed, pos)[0] != tok


def test_the_largest_noise_decides_a_close_draw_by_its_value() -> None:
    """The top uniform's noise, 0.01 short of a tie or past it (`largest_noise_rows`), decides the draw as 16.64 says.
    tests/kernels/test_kernels.py runs the same rows through the kernel."""
    assert run(largest_noise_rows(), 1.0, 1.0, 0, 0.0, 5738, 1) == [3980, 398]


def test_a_full_vocabulary_never_samples_a_token_far_below_the_top() -> None:
    """The plain group (temperature above 0, top-k and top-p off) over a 262,144-token vocabulary: one token at logit
    0 and the rest at -40, over 1,000 positions. The noise spans 19.49 (`gumbel`), so no off-peak token is sampled,
    though 14 of them draw the top uniform (a row has one with probability 1 - (1 - 2^-24)^V, 1.55%), whose noise
    would be +inf unclamped and win. tests/kernels/test_kernels.py runs the same rows through the kernel."""
    from llm_mini_infer.runner.cuda.sampler import U_MAX, gumbel

    v, n, chunk, peak = 262_144, 1000, 20, 12_345
    row = torch.full((1, v), -40.0)
    row[0, peak] = 0.0
    off = torch.arange(v) != peak
    top = -torch.log(-torch.log(torch.tensor(U_MAX)))
    toks, at_top = [], 0
    for p0 in range(1, n + 1, chunk):  # a few rows at a time: [chunk, V] temporaries
        pos = np.arange(p0, p0 + chunk)
        toks += run(row.expand(chunk, -1), 1.0, 1.0, 0, 0.0, 0, pos)
        noise = gumbel(row_keys(torch.zeros(chunk, dtype=torch.int64), torch.as_tensor(pos)), torch.arange(v))
        at_top += int(((noise == top) & off).sum())
    assert toks == [peak] * n
    assert at_top == 14


# -- the grammar-aware heads: padded id maps, masks tested through the ids -------------------------------------------

COMBOS = [  # temperature, top_p, top_k, min_p: every group of plan_sampling, alone and combined
    (0.0, 1.0, 0, 0.0), (1.0, 1.0, 0, 0.0), (0.6, 1.0, 0, 0.0), (1.0, 1.0, 20, 0.0), (1.0, 1.0, 1, 0.0),
    (1.0, 0.8, 0, 0.0), (1.0, 1.0, 0, 0.1), (0.7, 0.8, 20, 0.0), (1.3, 0.9, 50, 0.05), (1.0, 0.5, 0, 0.1),
    (0.9, 0.95, 3000, 0.0), (2.0, 0.999, 0, 0.0)]


def _params(n: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """Per-row parameters: COMBOS in turn, each row with its own seed and position."""
    c = np.array([COMBOS[i % len(COMBOS)] for i in range(n)])
    return (c[:, 0].astype(np.float32), c[:, 1].astype(np.float32), c[:, 2].astype(np.int64),
            c[:, 3].astype(np.float32), rng.integers(0, 1 << 62, n), rng.integers(0, 100_000, n))


def _ref(logits: torch.Tensor, vocab_ids: torch.Tensor, params: tuple[np.ndarray, ...],
         mask: torch.Tensor | None = None, mask_row: np.ndarray | None = None) -> tuple[list[int], list[int]]:
    t, p, k, m, seed, pos = params
    plan = plan_sampling(t, p, k, m, logits.shape[1])
    for g in plan.groups:
        g.idx = torch.as_tensor(g.rows)
    out, status = sample(logits, plan, torch.as_tensor(t), torch.as_tensor(p), torch.as_tensor(k), torch.as_tensor(m),
                         row_keys(torch.as_tensor(seed), torch.as_tensor(pos)), vocab_ids, mask=mask,
                         mask_row=None if mask_row is None else torch.as_tensor(mask_row))
    return out.tolist(), status.tolist()


def _pack(allowed: np.ndarray) -> torch.Tensor:
    """bool [m, V] -> int32 mask words [m, ceil(V / 32)], bit t % 32 of word t // 32 (the engine's layout)."""
    bits = np.pad(allowed, ((0, 0), (0, -allowed.shape[1] % 32)))
    return torch.as_tensor(np.packbits(bits, axis=1, bitorder="little").view(np.int32).copy())


def _padded(ids: np.ndarray, cap: int) -> torch.Tensor:
    """Ascending ids padded with -1 to `cap` (int64, as the runners pass `vocab_ids`)."""
    out = torch.full((cap,), -1, dtype=torch.int64)
    out[:len(ids)] = torch.as_tensor(ids)
    return out


def _columns(dense: torch.Tensor, vocab_ids: torch.Tensor) -> torch.Tensor:
    """The dense logits' columns at `vocab_ids`, as a head over those rows computes them. A padding id gathers row 0,
    whose logit is lifted here above every real one, so that a padding column taken for a candidate would show."""
    cols = dense.index_select(1, vocab_ids.clamp(min=0))
    cols[:, vocab_ids < 0] += 100.0
    return cols


def test_the_unions_columns_sample_as_the_dense_logits_under_the_same_masks() -> None:
    """The grammar-sparse head's exactness: it samples only the union of the step's masks (ascending ids, padded with
    -1), under the same masks tested through the ids, and gets the dense path's tokens and statuses exactly: every
    parameter group, BF16 ties at the top, masks shared by rows, empty masks (MASK_EMPTY), NaN rows (NONFINITE), with
    padding and (first round) without."""
    for it in range(16):
        rng = np.random.default_rng(100 + it)
        g = torch.Generator().manual_seed(it)
        v, n, n_masks = int(rng.choice([4096, 6144, 8192])), 4 * len(COMBOS), 16
        allowed = np.zeros((n_masks, v), bool)
        for i in range(n_masks):
            allowed[i, rng.choice(v, int(rng.integers(1, 65)), replace=False)] = True
        allowed[rng.choice(n_masks, 2, replace=False)] = False  # two masks allow nothing
        mask_row = rng.integers(0, n_masks, n)  # rows share masks
        m0 = int(np.flatnonzero(allowed.any(1))[0])
        allowed[m0, 0] = True  # token 0, whose head row a padding id gathers: its bit must not admit padding
        mask_row[:len(COMBOS)] = m0
        dense = torch.randn(n, v, generator=g) * torch.linspace(0.5, 8, n)[:, None]
        if it % 2:
            dense = dense.to(torch.bfloat16).float()  # the model's logits: many exact ties
        for r in range(0, n, 3):  # two allowed tokens tie at the top
            toks = np.flatnonzero(allowed[mask_row[r]])
            if len(toks) >= 2:
                dense[r, torch.as_tensor(rng.choice(toks, 2, replace=False))] = dense[r, toks].max() + 1.0
        union = np.flatnonzero(allowed.any(0))
        dense[n - 1] = float("nan")  # a NaN hidden state: every column
        dense[n - 2, union[0]] = float("nan")
        cap = len(union) if it == 0 else (len(union) // 128 + 1 + int(rng.integers(0, 3))) * 128
        vocab_ids = _padded(union, cap)
        params, mask = _params(n, rng), _pack(allowed)
        want = _ref(dense, torch.arange(v), params, mask, mask_row)
        got = _ref(_columns(dense, vocab_ids), vocab_ids, params, mask, mask_row)
        assert got == want, (it, [(r, want[0][r], got[0][r]) for r in range(n) if got[0][r] != want[0][r]][:5])
        tok, st = want
        assert st[n - 2:] == [NONFINITE, NONFINITE] and MASK_EMPTY in st
        ok = np.flatnonzero(np.asarray(st) == 0)
        assert allowed[mask_row[ok], np.asarray(tok)[ok]].all()
        assert len(set(np.asarray(tok)[ok].tolist())) > len(ok) // 2  # draws, not one token


def test_padded_ids_sample_only_their_tokens() -> None:
    """Padded ids without masks (the assistant's head over the grammar's universe): never a padding column nor a
    token outside the ids, exactly the dense logits' tokens under a mask that allows the ids alone. With masks over
    the whole vocabulary, a masked row samples its mask's tokens among the ids (other tokens' bits never matter: a
    mask of tokens outside them is MASK_EMPTY) and an unmasked row every id. Only padding: MASK_EMPTY. The NaN check
    reads the padding columns too (the last row is NaN, the one before only in token 0, which they gather)."""
    for it in range(8):
        rng = np.random.default_rng(200 + it)
        g = torch.Generator().manual_seed(50 + it)
        v, cap, n = 5000, 640, 2 * len(COMBOS)
        ids = np.sort(rng.choice(np.arange(1, v), int(rng.integers(1, cap)), replace=False))  # token 0 not among them
        in_ids = np.zeros(v, bool)
        in_ids[ids] = True
        dense = torch.randn(n, v, generator=g) * torch.linspace(0.5, 8, n)[:, None]
        if it % 2:
            dense = dense.to(torch.bfloat16).float()
        dense[n - 1] = float("nan")
        dense[n - 2, 0] = float("nan")
        vocab_ids = _padded(ids, cap)
        cols, params = _columns(dense, vocab_ids), _params(n, rng)
        got = _ref(cols, vocab_ids, params)
        assert got == _ref(dense, torch.arange(v), params, _pack(in_ids[None]), np.zeros(n, np.int64)), it
        tok, st = got
        assert st == [0] * (n - 2) + [NONFINITE] * 2 and in_ids[tok[:-2]].all()
        masks = np.zeros((4, v), bool)
        masks[0] = rng.random(v) < 0.5  # about half of the ids, and of the other tokens
        masks[1, rng.choice(ids, min(5, len(ids)), replace=False)] = True  # a few of the ids
        masks[2] = ~in_ids  # tokens outside the ids only
        masks[3, rng.choice(v, 40, replace=False)] = True
        mask_row = rng.integers(-1, len(masks), n)  # -1: unmasked
        mask_row[:2] = [2, -1]
        got = _ref(cols, vocab_ids, params, _pack(masks), mask_row)
        dense_masks = _pack(np.concatenate([masks & in_ids, in_ids[None]]))  # the last: the ids alone
        assert got == _ref(dense, torch.arange(v), params, dense_masks, np.where(mask_row < 0, len(masks), mask_row))
        tok, st = got
        assert st[:2] == [MASK_EMPTY, 0] and st[-2:] == [NONFINITE] * 2
        ok = np.flatnonzero(np.asarray(st) == 0)
        assert in_ids[np.asarray(tok)[ok]].all()
        tok, st = _ref(cols, _padded(ids[:0], cap), params)  # only padding
        assert tok == [0] * n and st == [MASK_EMPTY] * (n - 2) + [NONFINITE] * 2


# -- the sampled distribution: a goodness of fit ----------------------------------------------------------------------


def test_sampled_frequencies_fit_the_target_distribution() -> None:
    """Many draws of one row against the distribution it samples from, on every group of plan_sampling: the tokens
    excluded (by the mask, the largest logit's among them, and by min-p, top-k and top-p) never drawn, the others as
    often as softmax(logits / temperature) over the kept tokens says. tests/kernels/test_kernels.py draws the same rows
    through the kernel."""
    logits, allowed = fit_row()
    v, n = len(logits), FIT_DRAWS
    for temp, top_p, top_k, min_p, masked in FIT_CASES:
        params = (np.full(n, temp, np.float32), np.full(n, top_p, np.float32), np.full(n, top_k, np.int64),
                  np.full(n, min_p, np.float32), np.full(n, FIT_SEED, np.int64), np.arange(1, n + 1))
        mask, mask_row = (_pack(allowed[None]), np.zeros(n, np.int64)) if masked else (None, None)
        tok, st = _ref(torch.as_tensor(logits)[None].expand(n, -1), torch.arange(v), params, mask, mask_row)
        assert not any(st)
        probs = target_distribution(logits, temp, top_p, top_k, min_p, allowed if masked else np.ones(v, bool))
        assert_fits(np.asarray(tok), probs)
