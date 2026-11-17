// The GDN causal depthwise conv (width K = 4) over each row's (conv state ++ tokens), with SiLU, the state update and
// the conv states of mid-chunk checkpoints (models/qwen3_5/backend.py `GDN.conv`). Like the torch reference, it rounds
// each input × weight product to BF16, adds the products in fp32 in tap order, and rounds the SiLU output once.
//
// Each thread computes one channel over a tile of up to kTile tokens of one row, with the last K - 1 inputs in
// registers. Rows [0, n_dec) have one token (decode rows, one tile each). Longer rows split into tiles, so that a
// 2,048-token row runs on enough threads (one thread walking a row for its channel took 1 ms per layer on the 27B).
// Only the first tile of a row reads the old state and writes the checkpoints and the new state (after its tokens).
// Later tiles take their window from x. Rows need no synchronisation.
//
// MTP verify rows (kernels/ops.py `ConvMtp`): rows [0, n_mtp) are windows of `width` tokens (one tile).
// The live state first absorbs the first fold_n[req] inputs of the previous window of the slot, from `rec` (the fold
// of the GDN replay), and is written back. The kernel then convolves the window without absorbing it, and stores its
// inputs in `rec` for the next fold. Each thread owns its channel of the row, so `rec` needs no synchronisation.
//
// Programmatic dependent launch (common.cuh): the kernel waits for the kernel before it as its first statement, then
// lets the next kernel launch; the launcher adds the attribute while the library's flag is on. FOLD and checkpoints
// (outside the CUDA graphs) launch serialized.
#include "common.cuh"

constexpr int kW1 = 3;     // conv_kernel - 1
constexpr int kTile = 32;  // tokens per thread in multi-token rows (> kW1: only first tiles read the old state)

// x [T, C] at row stride `xs`; w [C, 4]; state: the layer's view of the conv pool, slot stride `ss`, [C, 3] per slot
// (oldest first). cu [R + 1] int64 token offsets of the rows; slot [R] int64 live state slot of each row. Checkpoints:
// ck_ptr [R + 1] int64 into ck_off/ck_dst (offset in the row, destination state slot); offsets outside [0, n) are
// ignored (the end of a chunk is the live state itself).
__global__ void conv_kernel(const bf16* __restrict__ x, int64_t xs, const bf16* __restrict__ w,
                            bf16* __restrict__ state, int64_t ss, const int64_t* __restrict__ cu,
                            const int64_t* __restrict__ slot, const int64_t* __restrict__ ck_ptr,
                            const int64_t* __restrict__ ck_off, const int64_t* __restrict__ ck_dst,
                            bf16* __restrict__ out, int C, int rows, int n_dec, int n_mtp, int width,
                            const int64_t* __restrict__ mtp_req, const int64_t* __restrict__ fold_n,
                            bf16* __restrict__ rec, int64_t rec_stride) {
  lmk_pdl_wait();  // PDL: nothing a predecessor writes is read before it (common.cuh)
  lmk_pdl_trigger();
  int r = blockIdx.y, j = 0;  // this block's row and tile in it
  if (r >= n_dec) {           // a multi-token row: scan the rows after the decode rows (few) for this tile
    int y = r - n_dec;
    for (r = n_dec; r < rows; ++r) {
      const int nt = (int)((cu[r + 1] - cu[r] + kTile - 1) / kTile);
      if (y < nt) break;
      y -= nt;
    }
    if (r >= rows) return;  // the grid is an upper bound on the tiles
    j = y;
  }
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const int64_t t0 = cu[r], t1 = cu[r + 1], n = t1 - t0;
  const int64_t ts = t0 + (int64_t)j * kTile, te = ts + kTile < t1 ? ts + kTile : t1;
  const bf16 w0 = w[c * 4 + 0], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
  bf16* st = state + slot[r] * ss + (int64_t)c * kW1;
  bf16 s0, s1, s2;
  bf16* rc = nullptr;  // this row's window records (MTP)
  if (r < n_mtp) {
    rc = rec + mtp_req[r] * rec_stride + (int64_t)c * width;
    int nf = (int)fold_n[mtp_req[r]];
    nf = nf < 0 ? 0 : (nf > width ? width : nf);
    s0 = st[0];
    s1 = st[1];
    s2 = st[2];
    for (int i = 0; i < nf; ++i) {  // absorb the previous window's committed inputs
      s0 = s1;
      s1 = s2;
      s2 = rc[i];
    }
    if (nf) {
      st[0] = s0;
      st[1] = s1;
      st[2] = s2;
    }
  } else if (j == 0) {
    s0 = st[0];
    s1 = st[1];
    s2 = st[2];
  } else {
    s0 = x[(ts - 3) * xs + c];
    s1 = x[(ts - 2) * xs + c];
    s2 = x[(ts - 1) * xs + c];
  }
  const bf16 o0 = s0, o1 = s1, o2 = s2;
  for (int64_t t = ts; t < te; ++t) {
    const bf16 xt = x[t * xs + c];
    float acc = bf2f(__hmul(s0, w0));
    acc = fadd(acc, bf2f(__hmul(s1, w1)));
    acc = fadd(acc, bf2f(__hmul(s2, w2)));
    acc = fadd(acc, bf2f(__hmul(xt, w3)));
    out[t * C + c] = f2bf(t_silu(acc));
    if (rc != nullptr) rc[t - ts] = xt;  // the window's inputs: the next fold's records
    s0 = s1;
    s1 = s2;
    s2 = xt;
  }
  if (j != 0 || rc != nullptr) return;
  // ext = old state ++ the row's inputs: the state after k tokens of the row is ext[k], ext[k + 1], ext[k + 2]
  auto ext = [&](int64_t i) -> bf16 { return i == 0 ? o0 : i == 1 ? o1 : i == 2 ? o2 : x[(t0 + i - 3) * xs + c]; };
  if (ck_ptr != nullptr) {
    for (int64_t k = ck_ptr[r]; k < ck_ptr[r + 1]; ++k) {
      const int64_t off = ck_off[k];
      if (off < 0 || off >= n) continue;
      bf16* d = state + ck_dst[k] * ss + (int64_t)c * kW1;
      d[0] = ext(off);
      d[1] = ext(off + 1);
      d[2] = ext(off + 2);
    }
  }
  st[0] = ext(n);
  st[1] = ext(n + 1);
  st[2] = ext(n + 2);
}

// rows R, tokens T. Rows [0, n_dec) have one token each (0 if unknown: the kernel then looks up every row). MTP: rows
// [0, n_mtp) are verify windows of `width` tokens (n_dec must be 0). mtp_req int64 [n_mtp]: their request slots.
// fold_n int64 [req slots]. rec: a layer's view [req slots, C, width] at request stride rec_stride.
LMK_API int lmk_causal_conv(const void* x, int64_t x_stride, const void* weight, void* state, int64_t state_stride,
                            const void* cu, const void* slot, const void* ck_ptr, const void* ck_off,
                            const void* ck_dst, void* out, int rows, int C, int tokens, int n_dec, int n_mtp,
                            int width, const void* mtp_req, const void* fold_n, void* rec, int64_t rec_stride,
                            cudaStream_t stream) {
  if (rows <= 0) return 0;
  if (n_mtp && (n_dec || width < 1 || width > kTile)) return -1;
  const int threads = 128;
  // An upper bound on the tiles: one per decode row, plus ceil(other tokens / kTile) in total, plus one per other row
  // (each row rounds up)
  const int tiles = n_dec + (tokens - n_dec + kTile - 1) / kTile + (rows - n_dec);
  dim3 grid((C + threads - 1) / threads, tiles);
  return lmk_launch(conv_kernel, grid, dim3(threads), 0, stream, (const bf16*)x, x_stride, (const bf16*)weight,
                    (bf16*)state, state_stride, (const int64_t*)cu, (const int64_t*)slot, (const int64_t*)ck_ptr,
                    (const int64_t*)ck_off, (const int64_t*)ck_dst, (bf16*)out, C, rows, n_dec, n_mtp, width,
                    (const int64_t*)mtp_req, (const int64_t*)fold_n, (bf16*)rec, rec_stride);
}

// FOLD and MTP decode checkpoints for the conv state of every GDN layer (kernels/ops.py `conv_fold`). state [slots,
// layers, C, 3] and rec [req slots, layers, C, width] are contiguous. Per row, the state of slot src absorbs the first
// n inputs of the window of request slot req, and the result goes to slot dst.
__global__ void conv_fold_kernel(bf16* __restrict__ state, const bf16* __restrict__ rec, const int64_t* __restrict__ req,
                                 const int64_t* __restrict__ src, const int64_t* __restrict__ dst,
                                 const int64_t* __restrict__ count, int layers, int C, int width) {
  const int row = blockIdx.y, layer = blockIdx.z;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  int n = (int)count[row];
  n = n < 0 ? 0 : (n > width ? width : n);
  const int64_t lc = ((int64_t)layer * C + c);
  const bf16* a = state + (src[row] * layers * C + lc) * kW1;
  bf16* d = state + (dst[row] * layers * C + lc) * kW1;
  const bf16* r = rec + (req[row] * layers * C + lc) * width;
  bf16 s0 = a[0], s1 = a[1], s2 = a[2];
  for (int i = 0; i < n; ++i) {
    s0 = s1;
    s1 = s2;
    s2 = r[i];
  }
  d[0] = s0;
  d[1] = s1;
  d[2] = s2;
}

LMK_API int lmk_conv_fold(void* state, const void* rec, const void* req, const void* src, const void* dst,
                          const void* n, int rows, int layers, int C, int width, cudaStream_t stream) {
  if (rows <= 0) return 0;
  const int threads = 128;
  dim3 grid((C + threads - 1) / threads, rows, layers);
  conv_fold_kernel<<<grid, threads, 0, stream>>>((bf16*)state, (const bf16*)rec, (const int64_t*)req,
                                                 (const int64_t*)src, (const int64_t*)dst, (const int64_t*)n, layers,
                                                 C, width);
  return lmk_status();
}
