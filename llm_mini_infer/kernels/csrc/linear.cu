// BF16 linear layer for few rows: out[m, n] = bf16(sum_k x[m, k] * w[n, k]), fp32 accumulation. It serves thin
// weights at the small M of decode and verify steps, such as the GDN layers' in_proj_ba (N = 96, K = 5,120 on
// Qwen3.8-27B). There cuBLAS is 7x slower at 2-4 rows than at 1 or 8 rows (RTX 5090: 29.8 us per call against 4.1
// and 4.9); an MTP verify window at batch 1 is 4 rows. cuBLAS keeps every other M.
//
// One CTA per weight row. Its threads split K in chunks of 8 elements, with 16-byte loads of the weight row and of
// each input row. So every load of the row is in flight at once. Each thread accumulates its chunks per input row with
// fp32 FMAs in k order. The CTA then sums the threads: warp sums first, then the warps in order. Each output is
// computed the same way at any M up to kMaxRows, so a row's result does not depend on the batch.
//
// Programmatic dependent launch (PDL, common.cuh): the kernel waits for its predecessor first, then lets the next
// kernel launch. The launcher adds the attribute while the library's flag is on (the runner's `dev.runner.cuda.pdl`).
#include "common.cuh"

namespace {

constexpr int kThreads = 128;
constexpr int kMaxRows = 8;

__device__ __forceinline__ void bf16x8(const uint4& v, float* f) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 t = __bfloat1622float2(p[j]);
    f[2 * j] = t.x;
    f[2 * j + 1] = t.y;
  }
}

template <int MT>
__global__ void __launch_bounds__(kThreads) linear_small_kernel(const bf16* __restrict__ x, int64_t xs,
                                                               const bf16* __restrict__ w, int64_t ws,
                                                               bf16* __restrict__ out, int64_t os, int M, int K) {
  lmk_pdl_wait();  // PDL: wait for the predecessor (common.cuh)
  lmk_pdl_trigger();
  __shared__ float red[MT][kThreads / 32];
  const int n = blockIdx.x;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  float acc[MT];
#pragma unroll
  for (int i = 0; i < MT; ++i) acc[i] = 0.0f;
  const bf16* wr = w + (int64_t)n * ws;
#pragma unroll 4
  for (int k = threadIdx.x * 8; k < K; k += kThreads * 8) {
    float wf[8];
    bf16x8(__ldg(reinterpret_cast<const uint4*>(wr + k)), wf);
#pragma unroll
    for (int i = 0; i < MT; ++i) {
      if (i < M) {
        float xf[8];
        bf16x8(__ldg(reinterpret_cast<const uint4*>(x + (int64_t)i * xs + k)), xf);
        float s = acc[i];
#pragma unroll
        for (int j = 0; j < 8; ++j) s = fmaf(xf[j], wf[j], s);
        acc[i] = s;
      }
    }
  }
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    const float v = warp_sum(acc[i]);
    if (lane == 0) red[i][warp] = v;
  }
  __syncthreads();
  if (threadIdx.x < M) {
    const int i = threadIdx.x;
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < kThreads / 32; ++j) s += red[i][j];
    out[(int64_t)i * os + n] = f2bf(s);
  }
}

template <int MT>
int launch(const bf16* x, int64_t xs, const bf16* w, int64_t ws, bf16* out, int64_t os, int M, int N, int K,
           cudaStream_t stream) {
  return lmk_launch(linear_small_kernel<MT>, dim3(N), dim3(kThreads), 0, stream, x, xs, w, ws, out, os, M, K);
}

}  // namespace

// x [M, K] at row stride xs, w [N, K] at row stride ws (elements), out [M, N] at row stride os. Requires M <= 8,
// K, xs and ws multiples of 8, and 16-byte-aligned x and w. Returns -1 otherwise.
LMK_API int lmk_linear_small(const void* x, int64_t xs, const void* w, int64_t ws, void* out, int64_t os, int M, int N,
                             int K, cudaStream_t stream) {
  if (M <= 0 || N <= 0) return 0;
  if (M > kMaxRows || K % 8 || xs % 8 || ws % 8 || ((uintptr_t)x & 15) || ((uintptr_t)w & 15)) return -1;
  const bf16* xp = (const bf16*)x;
  const bf16* wp = (const bf16*)w;
  bf16* op = (bf16*)out;
  if (M <= 1) return launch<1>(xp, xs, wp, ws, op, os, M, N, K, stream);
  if (M <= 2) return launch<2>(xp, xs, wp, ws, op, os, M, N, K, stream);
  if (M <= 4) return launch<4>(xp, xs, wp, ws, op, os, M, N, K, stream);
  return launch<8>(xp, xs, wp, ws, op, os, M, N, K, stream);
}

// BF16 linear layer whose row bits do not depend on M (batch invariance). It serves the GDN layers' in_proj_ba, also
// in Qwen3.8's FP8 checkpoints: out[m, n] = bf16(sum_k x[m, k] * w[n, k]) on tensor cores (mma.sync m16n8k16, fp32
// accumulation). K splits into S slices of K / S, with S = 8, 4, 2 or 1. S is the largest value that keeps whole
// 64-element stages, so it depends on K alone. A CTA computes a 32-row by 32-output tile over one slice, in
// 64-element stages.
// x and w go through shared memory, double-buffered with cp.async; rows and outputs past the matrix are zero-filled.
// Each of the 4 warps computes a 16 x 16 quarter of the tile, its k-steps in order. The CTA writes the tile's fp32
// partial sums. A second kernel adds each output's S partials in slice order and rounds once. A row's result depends
// on its own inputs, the weight and K only: not on M, on its tile, or on its place in the tile.
namespace {

constexpr int kRowsT = 32;  // rows and outputs per tile
constexpr int kStage = 64;  // K elements per stage
constexpr int kChunks = kStage / 8;  // 16-byte chunks per stage row

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// Byte offset of 16-byte chunk c of row r in a [32][kChunks] stage, XOR-swizzled by r % 8.
__device__ __forceinline__ int swz8(int r, int c) { return (r * kChunks + (c ^ (r & 7))) * 16; }

__device__ __forceinline__ void cp16(uint32_t dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void ldsm4(uint32_t addr, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Grid (row tiles, output tiles, S); part is fp32 [S, M, N].
__global__ void __launch_bounds__(128) linear_rows_kernel(const bf16* __restrict__ x, int64_t xs,
                                                          const bf16* __restrict__ w, int64_t ws,
                                                          float* __restrict__ part, int M, int N, int slice) {
  lmk_pdl_wait();  // PDL: wait for the predecessor (common.cuh)
  lmk_pdl_trigger();
  __shared__ alignas(128) unsigned char sx[2][kRowsT * kStage * 2];
  __shared__ alignas(128) unsigned char sw[2][kRowsT * kStage * 2];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int m0 = blockIdx.x * kRowsT, n0 = blockIdx.y * kRowsT, k0 = blockIdx.z * slice;
  const int stages = slice / kStage;
  auto load = [&](int st, int buf) {
    const int kb = k0 + st * kStage;
    for (int c = tid; c < 2 * kRowsT * kChunks; c += 128) {
      const int which = c / (kRowsT * kChunks), r = (c / kChunks) % kRowsT, ch = c % kChunks;
      if (which == 0) {
        const bool ok = m0 + r < M;
        cp16(smem_addr(sx[buf] + swz8(r, ch)), ok ? x + (int64_t)(m0 + r) * xs + kb + ch * 8 : x, ok);
      } else {
        const bool ok = n0 + r < N;
        cp16(smem_addr(sw[buf] + swz8(r, ch)), ok ? w + (int64_t)(n0 + r) * ws + kb + ch * 8 : w, ok);
      }
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };
  const int wr = 16 * (warp >> 1), wn = 16 * (warp & 1);  // this warp's quarter of the tile
  float c[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
  load(0, 0);
  for (int st = 0; st < stages; ++st) {
    const int buf = st & 1;
    if (st + 1 < stages) {
      load(st + 1, buf ^ 1);
      asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    } else {
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < kStage / 16; ++kk) {  // k-steps in order
      uint32_t a[4], b[4];
      ldsm4(smem_addr(sx[buf] + swz8(wr + (lane & 7) + 8 * ((lane >> 3) & 1), 2 * kk + (lane >> 4))), a);
      ldsm4(smem_addr(sw[buf] + swz8(wn + (lane & 7) + 8 * (lane >> 4), 2 * kk + ((lane >> 3) & 1))), b);
      mma16816(c[0], a, b[0], b[1]);
      mma16816(c[1], a, b[2], b[3]);
    }
    __syncthreads();  // the next iteration reloads this buffer
  }
  const int g = lane >> 2, t4 = lane & 3;
  float* dst = part + (int64_t)blockIdx.z * M * N;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int m = m0 + wr + g + 8 * (e >> 1), n = n0 + wn + 8 * j + 2 * t4 + (e & 1);
      if (m < M && n < N) dst[(int64_t)m * N + n] = c[j][e];
    }
  }
}

// out[m, n] = bf16(part[0][m][n] + part[1][m][n] + ... + part[S - 1][m][n]), added in slice order.
__global__ void linear_rows_sum_kernel(const float* __restrict__ part, bf16* __restrict__ out, int64_t os, int M,
                                       int N, int S) {
  lmk_pdl_wait();  // PDL: wait for the predecessor (common.cuh)
  lmk_pdl_trigger();
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x, mn = (int64_t)M * N;
  if (i >= mn) return;
  float s = 0.0f;
  for (int j = 0; j < S; ++j) s += part[j * mn + i];
  out[(i / N) * os + i % N] = f2bf(s);
}

}  // namespace

// x [M, K] at row stride xs, w [N, K] at row stride ws (elements), out [M, N] at row stride os. part is fp32
// [S, M, N] scratch. S must be the largest of 8, 4, 2 or 1 that divides K into whole 64-element stages. Returns -1
// for another S, a K that is not a multiple of 64, or unaligned rows.
LMK_API int lmk_linear_rows(const void* x, int64_t xs, const void* w, int64_t ws, void* out, int64_t os, void* part,
                            int M, int N, int K, int S, cudaStream_t stream) {
  if (M <= 0 || N <= 0) return 0;
  int want = 8;
  while (want > 1 && K % (want * kStage)) want >>= 1;
  if (K <= 0 || K % kStage || S != want || xs % 8 || ws % 8 || ((uintptr_t)x & 15) || ((uintptr_t)w & 15)) return -1;
  const int st = lmk_launch(linear_rows_kernel, dim3((M + kRowsT - 1) / kRowsT, (N + kRowsT - 1) / kRowsT, S),
                            dim3(128), 0, stream, (const bf16*)x, xs, (const bf16*)w, ws, (float*)part, M, N, K / S);
  if (st) return st;
  const int64_t mn = (int64_t)M * N;
  return lmk_launch(linear_rows_sum_kernel, dim3((unsigned)((mn + 255) / 256)), dim3(256), 0, stream,
                    (const float*)part, (bf16*)out, os, M, N, S);
}
