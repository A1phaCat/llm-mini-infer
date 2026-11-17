// The attention layer's input side in one kernel: per-head Gemma RMSNorm of q and k, partial NeoX RoPE on the first
// `rot` dims, and the paged KV write (models/qwen3_5: `Attention.forward`, `ForwardContext.qk_rope_store`). Like the
// torch reference, the norm rounds to BF16, and RoPE runs in fp32 against the BF16 cos/sin cache and rounds once.
//
// Programmatic dependent launch (common.cuh): the kernel waits for its predecessor as its first statement. It lets the
// next kernel launch only while the library's flag is on (`kTrigger`). FlashInfer's attention follows and
// launches with PDL by default. The launcher adds the attribute while the flag is on.
#include "common.cuh"

constexpr int kMaxHeadDim = 256;

// One warp per (token, head) over the H q heads, then the Hkv k heads. qkv [T, *] has row stride `qs`. The q heads sit
// at head stride `qhs` from offset 0 (2-D when q and the output gate interleave). The k heads start at `k_off` and the
// v heads at `v_off`, both with head stride D. q_out is [T, H, D] contiguous. The KV pool layer view has page stride
// `ps_kv`, then [2, page_size, Hkv, D] contiguous. kv_page and kv_off are [T] int64. cos and sin are [max_len, rot/2]
// BF16.
//
// Positions are [T] int64 (`pstride` 0) or, for interleaved M-RoPE (images), the (t, h, w) components `pstride` apart.
// Frequency j reads h when j % 3 == 1 and j < 3 * sec_h, w when j % 3 == 2 and j < 3 * sec_w, and t otherwise
// (`ops.mrope_components`).
template <bool kTrigger>
__global__ void qk_norm_rope_kernel(const bf16* __restrict__ qkv, int64_t qs, int64_t qhs, int64_t k_off,
                                    int64_t v_off, const bf16* __restrict__ qw, const bf16* __restrict__ kw,
                                    const int64_t* __restrict__ pos, int64_t pstride, int sec_h, int sec_w,
                                    const bf16* __restrict__ cos,
                                    const bf16* __restrict__ sin, bf16* __restrict__ q_out,
                                    bf16* __restrict__ kv, int64_t ps_kv, int page_size,
                                    const int64_t* __restrict__ kv_page, const int64_t* __restrict__ kv_off,
                                    int64_t T, int H, int Hkv, int D, int rot, float eps) {
  lmk_pdl_wait();  // PDL: read nothing before the predecessor finishes (common.cuh)
  if (kTrigger) lmk_pdl_trigger();
  __shared__ float buf[8][kMaxHeadDim];
  const int wib = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int64_t item = (int64_t)blockIdx.x * (blockDim.x >> 5) + wib;
  const int64_t per_token = H + Hkv;
  if (item >= T * per_token) return;
  const int64_t t = item / per_token;
  const int hh = (int)(item % per_token);
  const bool is_q = hh < H;
  const int h = is_q ? hh : hh - H;
  const bf16* src = qkv + t * qs + (is_q ? (int64_t)h * qhs : k_off + (int64_t)h * D);
  const bf16* w = is_q ? qw : kw;
  const int per = D / 32;
  float v[kMaxHeadDim / 32];
  float ss = 0.0f;
#pragma unroll
  for (int k = 0; k < kMaxHeadDim / 32; ++k) {
    if (k < per) {
      v[k] = bf2f(src[lane + 32 * k]);
      ss = fadd(ss, fmul(v[k], v[k]));
    }
  }
  ss = warp_sum(ss);
  const float inv = rsqrtf(fadd(fmul(ss, 1.0f / (float)D), eps));
  float* b = buf[wib];
#pragma unroll
  for (int k = 0; k < kMaxHeadDim / 32; ++k) {
    if (k < per) {
      const int i = lane + 32 * k;
      b[i] = bf2f(f2bf(fmul(fmul(v[k], inv), fadd(1.0f, bf2f(w[i])))));  // the normed value, rounded to BF16
    }
  }
  __syncwarp();
  const int half = rot / 2;
  const int64_t p0 = pos[t], p1 = pos[pstride + t], p2 = pos[2 * pstride + t];
  bf16* dst;
  if (is_q) {
    dst = q_out + (t * H + h) * D;
  } else {
    const int64_t off = kv_off[t];
    dst = kv + kv_page[t] * ps_kv + (off * Hkv + h) * D;  // K plane; the V plane is page_size * Hkv * D further
  }
#pragma unroll
  for (int k = 0; k < kMaxHeadDim / 32; ++k) {
    if (k < per) {
      const int i = lane + 32 * k;
      float o = b[i];
      if (i < rot) {
        const int j = i < half ? i : i - half;
        const int64_t p = (j % 3 == 1 && j < 3 * sec_h) ? p1 : (j % 3 == 2 && j < 3 * sec_w) ? p2 : p0;
        const float c = bf2f(cos[p * half + j]), s = bf2f(sin[p * half + j]);
        o = i < half ? fsub(fmul(b[i], c), fmul(b[i + half], s)) : fadd(fmul(b[i], c), fmul(b[i - half], s));
      }
      dst[i] = f2bf(o);
    }
  }
  if (!is_q) {  // V: copied unchanged into the V plane
    const bf16* vs = qkv + t * qs + v_off + (int64_t)h * D;
    bf16* vd = dst + (int64_t)page_size * Hkv * D;
    for (int i = lane; i < D; i += 32) vd[i] = vs[i];
  }
}

LMK_API int lmk_qk_norm_rope(const void* qkv, int64_t qkv_stride, int64_t q_head_stride, int64_t k_off, int64_t v_off,
                             const void* q_weight, const void* k_weight, const void* positions, int64_t pstride,
                             int sec_h, int sec_w, const void* cos, const void* sin, void* q_out, void* kv_layer,
                             int64_t kv_page_stride, int page_size, const void* kv_page, const void* kv_off,
                             int64_t T, int H, int Hkv, int D, int rot, float eps, cudaStream_t stream) {
  if (T <= 0) return 0;
  if (D % 32 || D > kMaxHeadDim || rot > D || rot % 2) return -1;
  const int64_t n = T * (H + Hkv);
  return lmk_launch(lmk_pdl_flag() ? qk_norm_rope_kernel<true> : qk_norm_rope_kernel<false>,
                    dim3((unsigned)((n + 7) / 8)), dim3(256), 0, stream, (const bf16*)qkv, qkv_stride, q_head_stride,
                    k_off, v_off, (const bf16*)q_weight, (const bf16*)k_weight, (const int64_t*)positions, pstride,
                    sec_h, sec_w, (const bf16*)cos, (const bf16*)sin, (bf16*)q_out, (bf16*)kv_layer, kv_page_stride,
                    page_size, (const int64_t*)kv_page, (const int64_t*)kv_off, T, H, Hkv, D, rot, eps);
}
