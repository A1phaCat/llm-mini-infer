/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * GDN rollback by replay for MTP verification (docs/architecture.md, "Speculative decoding"). It follows
 * SGLang 0.5.22's ReplaySSM spec verify (python/sglang/kernels/ops/attention/fla/gdn_replayssm_spec_decode.py),
 * staged as "fold at every commit", and is written in CUDA on the arithmetic of vLLM's fused GDN decode kernel
 * (gdn_decode.cu, vendored): its helpers and its delta-rule update.
 *
 * gdn_mtp_kernel: one block per (verify row, value head). A verify row is a window of L = 1 + k tokens (the pending
 * token and k drafts). The live state first replays the first fold_n[req] tokens of the previous window of the slot,
 * from their records (the kernel's own inputs: raw k, v, a, b per value head). The kernel writes it back: the fold.
 * It then computes the window outputs from the folded state in registers and never writes the window states: a rejected
 * draft costs no rollback, and a step reads and writes the state once, as a plain decode step. The window inputs
 * become the records for the next fold. A row whose state slot is <= 0 (padding) writes zero outputs and nothing else.
 *
 * gdn_fold_kernel: FOLD and the MTP decode checkpoints. For every GDN layer, the state of slot src replays the first
 * n records of request slot req and goes to slot dst (in place when dst == src).
 *
 * Both kernels replay a record with the arithmetic that computed it, in the warp layout of gdn_decode.cu (four chunks
 * of 32 V rows, 4 rows per warp, 4 K values per lane). A folded state is bit-identical to the window state in
 * registers after those tokens.
 *
 * Changes for llm-mini-infer besides the replay: gdn_mtp_kernel uses programmatic dependent launch (common.cuh): it
 * waits for the kernel before it as its first statement, before any memory access, then lets the next kernel launch.
 * gdn_fold_kernel (outside the CUDA graphs) launches serialized.
 */
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "common.cuh"  // llm-mini-infer: LMK_API and the PDL helpers

namespace {

constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kChunkV = 32;
constexpr int kNumChunks = kDimV / kChunkV;
constexpr int kRowsPerWarp = kChunkV / kWarps;
constexpr int kMaxWin = 4;           // 1 + k: k <= 3
constexpr int kMaxTok = 2 * kMaxWin;  // fold + window, one warp each
constexpr int kDtBiasBFloat16 = 1;
constexpr int kDtBiasFloat16 = 2;
constexpr float kFp8Max = 448.0f;
constexpr float kQuantEps = 1e-10f;

template <typename StateT>
__device__ __forceinline__ void cp_async_16b(StateT* smem_ptr, const StateT* gmem_ptr) {
  const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }

template <typename StateT>
__device__ __forceinline__ void copy_state_chunk(StateT* shared_state, const StateT* state, int chunk, int thread) {
  constexpr int kElementsPerCopy = 16 / sizeof(StateT);
  constexpr int kCopiesPerChunk = kChunkV * kDimK / kElementsPerCopy;
  const int stage = chunk % 2;
  for (int copy = thread; copy < kCopiesPerChunk; copy += kThreads) {
    const int element = copy * kElementsPerCopy;
    cp_async_16b(shared_state + stage * kChunkV * kDimK + element, state + chunk * kChunkV * kDimK + element);
  }
  cp_async_commit();
}

template <typename StateT>
__device__ __forceinline__ float4 load_state4(const StateT* state);
template <>
__device__ __forceinline__ float4 load_state4<float>(const float* state) {
  return *reinterpret_cast<const float4*>(state);
}
template <>
__device__ __forceinline__ float4 load_state4<__nv_bfloat16>(const __nv_bfloat16* state) {
  const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(state);
  const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(state + 2);
  return make_float4(__bfloat162float(lo.x), __bfloat162float(lo.y), __bfloat162float(hi.x), __bfloat162float(hi.y));
}
template <typename StateT>
__device__ __forceinline__ void store_state4(StateT* state, float4 value);
template <>
__device__ __forceinline__ void store_state4<float>(float* state, float4 value) {
  *reinterpret_cast<float4*>(state) = value;
}
template <>
__device__ __forceinline__ void store_state4<__nv_bfloat16>(__nv_bfloat16* state, float4 value) {
  *reinterpret_cast<__nv_bfloat162*>(state) = __floats2bfloat162_rn(value.x, value.y);
  *reinterpret_cast<__nv_bfloat162*>(state + 2) = __floats2bfloat162_rn(value.z, value.w);
}

__device__ __forceinline__ float sigmoid_fast(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float silu_fast(float x) { return x * sigmoid_fast(x); }
__device__ __forceinline__ float softplus_fast(float x) { return x > 20.0f ? x : log1pf(__expf(x)); }

__device__ __forceinline__ float load_dt_bias(const void* dt_bias, int head, int dt_bias_type) {
  if (dt_bias_type == kDtBiasBFloat16) return __bfloat162float(static_cast<const __nv_bfloat16*>(dt_bias)[head]);
  if (dt_bias_type == kDtBiasFloat16) return __half2float(static_cast<const __half*>(dt_bias)[head]);
  return static_cast<const float*>(dt_bias)[head];
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset);
  return value;
}

struct Sum2 {
  float x;
  float y;
};

__device__ __forceinline__ Sum2 warp_reduce_sum_pair(float x, float y) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffffu, x, offset);
    y += __shfl_xor_sync(0xffffffffu, y, offset);
  }
  return {x, y};
}

// The token inputs in shared memory: one warp per token, as gdn_decode.cu loads them. k (and q) are L2-normalised
// from their raw BF16 values; decay = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b).
struct TokenSmem {
  float q[kMaxWin][kDimK];
  float k[kMaxTok][kDimK];
  __nv_bfloat16 v[kMaxTok][kDimV];
  __nv_bfloat16 out[kMaxWin][kDimV];
  float decay[kMaxTok];
  float beta[kMaxTok];
};

__device__ __forceinline__ void load_token(TokenSmem& sm, int slot_t, int lane, const __nv_bfloat16* k_raw,
                                           const __nv_bfloat16* v_raw, const __nv_bfloat16* q_raw, float scale,
                                           float a_value, float b_value, float a_log, float dt_bias, int q_slot) {
  float q_values[4] = {0.f, 0.f, 0.f, 0.f};
  float k_values[4];
  float q_square = 0.0f;
  float k_square = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int dim = lane + i * 32;
    if (q_raw != nullptr) q_values[i] = __bfloat162float(q_raw[dim]);
    k_values[i] = __bfloat162float(k_raw[dim]);
    sm.v[slot_t][dim] = v_raw[dim];
    q_square += q_values[i] * q_values[i];
    k_square += k_values[i] * k_values[i];
  }
  const Sum2 qk_sums = warp_reduce_sum_pair(q_square, k_square);
  const float q_scale = __shfl_sync(0xffffffffu, lane == 0 ? rsqrtf(qk_sums.x + 1.0e-6f) * scale : 0.0f, 0);
  const float k_scale = __shfl_sync(0xffffffffu, lane == 0 ? rsqrtf(qk_sums.y + 1.0e-6f) : 0.0f, 0);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int dim = lane + i * 32;
    if (q_raw != nullptr) sm.q[q_slot][dim] = q_values[i] * q_scale;
    sm.k[slot_t][dim] = k_values[i] * k_scale;
  }
  if (lane == 0) {
    const float g = -__expf(a_log) * softplus_fast(a_value + dt_bias);
    sm.decay[slot_t] = __expf(g);
    sm.beta[slot_t] = sigmoid_fast(b_value);
  }
}

// One token of the gated delta rule on the 4 rows of a warp in the current V chunk (gdn_decode.cu's, unchanged). With
// `with_out`, lane 0 writes the output h q.
__device__ __forceinline__ void update(float (&h)[kRowsPerWarp][4], const TokenSmem& sm, int t, int k_base,
                                       int chunk, const int (&rows)[kRowsPerWarp], int lane, bool with_out, int q_t,
                                       __nv_bfloat16* out_row) {
  const float4 k4 = *reinterpret_cast<const float4*>(&sm.k[t][k_base]);
  const float k_values[4] = {k4.x, k4.y, k4.z, k4.w};
  float dot_hk[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      h[row][i] *= sm.decay[t];
      dot_hk[row] += h[row][i] * k_values[i];
    }
  }
  const Sum2 dot_hk_01 = warp_reduce_sum_pair(dot_hk[0], dot_hk[1]);
  const Sum2 dot_hk_23 = warp_reduce_sum_pair(dot_hk[2], dot_hk[3]);
  const float reduced_hk[kRowsPerWarp] = {dot_hk_01.x, dot_hk_01.y, dot_hk_23.x, dot_hk_23.y};
  float q_values[4] = {0.f, 0.f, 0.f, 0.f};
  if (with_out) {
    const float4 q4 = *reinterpret_cast<const float4*>(&sm.q[q_t][k_base]);
    q_values[0] = q4.x;
    q_values[1] = q4.y;
    q_values[2] = q4.z;
    q_values[3] = q4.w;
  }
  float dot_hq[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) {
    const int value = chunk * kChunkV + rows[row];
    const float delta = (__bfloat162float(sm.v[t][value]) - reduced_hk[row]) * sm.beta[t];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      h[row][i] += k_values[i] * delta;
      dot_hq[row] += h[row][i] * q_values[i];
    }
  }
  if (with_out) {
    const Sum2 dot_hq_01 = warp_reduce_sum_pair(dot_hq[0], dot_hq[1]);
    const Sum2 dot_hq_23 = warp_reduce_sum_pair(dot_hq[2], dot_hq[3]);
    if (lane == 0) {
      out_row[chunk * kChunkV + rows[0]] = __float2bfloat16(dot_hq_01.x);
      out_row[chunk * kChunkV + rows[1]] = __float2bfloat16(dot_hq_01.y);
      out_row[chunk * kChunkV + rows[2]] = __float2bfloat16(dot_hq_23.x);
      out_row[chunk * kChunkV + rows[3]] = __float2bfloat16(dot_hq_23.y);
    }
  }
}

template <typename StateT>
__device__ __forceinline__ void store_rows(StateT* dst, const float (&h)[kRowsPerWarp][4], int chunk,
                                           const int (&rows)[kRowsPerWarp], int k_base) {
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) {
    const int value = chunk * kChunkV + rows[row];
    store_state4(dst + value * kDimK + k_base, make_float4(h[row][0], h[row][1], h[row][2], h[row][3]));
  }
}

struct Quant {
  uint8_t* q;  // [T, HV * 128] e4m3 or null
  float* s;    // [HV, s_ld]
  int64_t s_ld;
};

struct Records {
  __nv_bfloat16* k;   // [req slots, (layers,) HV, width, 128]
  __nv_bfloat16* v;
  __nv_bfloat16* ab;  // [req slots, (layers,) HV, width, 2]
  int64_t req_stride;  // elements between request slots (k and v; for ab: req_stride / 64)
  int width;
};

template <typename StateT, int ValueHeadsPerKeyHead, bool SigmoidGate>
__global__ __launch_bounds__(kThreads, 2) void gdn_mtp_kernel(
    const __nv_bfloat16* __restrict__ mixed_qkv, int64_t mixed_row, const __nv_bfloat16* __restrict__ a,
    int64_t a_row, const __nv_bfloat16* __restrict__ b, int64_t b_row, const float* __restrict__ a_log,
    const void* __restrict__ dt_bias, int dt_bias_type, const int* __restrict__ state_slots,
    const int* __restrict__ req_slots, const int* __restrict__ fold_n, StateT* __restrict__ state,
    int64_t state_slot_stride, const __nv_bfloat16* __restrict__ output_gate, int64_t gate_row,
    const void* __restrict__ norm_weight, bool norm_weight_is_bf16, __nv_bfloat16* __restrict__ out, Quant quant,
    Records rec, int H, int HV, float scale, float norm_eps) {
  lmk_pdl_wait();  // llm-mini-infer, PDL: nothing a predecessor writes is read or written before this (common.cuh)
  lmk_pdl_trigger();
  const int row = blockIdx.x;
  const int value_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int width = rec.width;
  const int bos = row * width;
  const int ss = state_slots[row];
  if (ss <= 0) {
    for (int linear = tid; linear < width * kDimV; linear += kThreads) {
      const int token = bos + linear / kDimV;
      const int value = linear % kDimV;
      const int64_t out_offset = (static_cast<int64_t>(token) * HV + value_head) * kDimV + value;
      if (out != nullptr) out[out_offset] = __float2bfloat16(0.0f);
      if (quant.q != nullptr) {
        quant.q[out_offset] = 0;
        if (value == 0) quant.s[value_head * quant.s_ld + token] = __fdiv_rn(kQuantEps, kFp8Max);
      }
    }
    return;
  }
  const int req = req_slots[row];
  int nf = fold_n[req];
  nf = nf < 0 ? 0 : (nf > width ? width : nf);
  const int total = nf + width;
  const int key_head = value_head / ValueHeadsPerKeyHead;
  __shared__ StateT shared_state[2][kChunkV][kDimK];
  __shared__ TokenSmem sm;

  StateT* live = state + static_cast<int64_t>(ss) * state_slot_stride + value_head * kDimV * kDimK;
  copy_state_chunk<StateT>(&shared_state[0][0][0], live, 0, tid);

  const int64_t rbase = static_cast<int64_t>(req) * rec.req_stride + static_cast<int64_t>(value_head) * width * 128;
  const int64_t abase = static_cast<int64_t>(req) * (rec.req_stride / 64) + static_cast<int64_t>(value_head) * width * 2;
  const float a_log_h = a_log[value_head];
  const float dt_h = load_dt_bias(dt_bias, value_head, dt_bias_type);
  if (warp < total) {
    if (warp < nf) {  // a record of the commit of the previous window
      const int t = warp;
      load_token(sm, t, lane, rec.k + rbase + t * 128, rec.v + rbase + t * 128, nullptr, scale,
                 __bfloat162float(rec.ab[abase + t * 2]), __bfloat162float(rec.ab[abase + t * 2 + 1]), a_log_h, dt_h,
                 0);
    } else {  // a window token
      const int t = warp - nf;
      const int token = bos + t;
      const int64_t base = static_cast<int64_t>(token) * mixed_row;
      load_token(sm, warp, lane, mixed_qkv + base + H * kDimK + key_head * kDimK,
                 mixed_qkv + base + 2 * H * kDimK + value_head * kDimV, mixed_qkv + base + key_head * kDimK, scale,
                 __bfloat162float(a[static_cast<int64_t>(token) * a_row + value_head]),
                 __bfloat162float(b[static_cast<int64_t>(token) * b_row + value_head]), a_log_h, dt_h, t);
    }
  }
  __syncthreads();

  const int k_base = lane * 4;
  int rows[kRowsPerWarp];
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) rows[r] = warp + r * kWarps;

#pragma unroll
  for (int chunk = 0; chunk < kNumChunks; ++chunk) {
    cp_async_wait_all();
    __syncthreads();
    if (chunk + 1 < kNumChunks) copy_state_chunk<StateT>(&shared_state[0][0][0], live, chunk + 1, tid);
    float h[kRowsPerWarp][4];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      const float4 sv = load_state4(&shared_state[chunk & 1][rows[r]][k_base]);
      h[r][0] = sv.x;
      h[r][1] = sv.y;
      h[r][2] = sv.z;
      h[r][3] = sv.w;
    }
    for (int t = 0; t < nf; ++t) update(h, sm, t, k_base, chunk, rows, lane, false, 0, nullptr);
    if (nf > 0) store_rows(live, h, chunk, rows, k_base);  // the fold: the commit, written once
    for (int t = nf; t < total; ++t) update(h, sm, t, k_base, chunk, rows, lane, true, t - nf, sm.out[t - nf]);
  }
  __syncthreads();

  if (warp < width) {  // the gated RMSNorm of each window output (gdn_decode.cu's epilogue), then the records
    const int t = warp;
    const int token = bos + t;
    float output_values[4];
    float sum_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      output_values[i] = __bfloat162float(sm.out[t][lane + i * 32]);
      sum_square += output_values[i] * output_values[i];
    }
    sum_square = warp_reduce_sum(sum_square);
    const float rstd = rsqrtf(sum_square / static_cast<float>(kDimV) + norm_eps);
    float rounded[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      const float gate_input =
          __bfloat162float(output_gate[static_cast<int64_t>(token) * gate_row + value_head * kDimV + value]);
      const float gate = SigmoidGate ? sigmoid_fast(gate_input) : silu_fast(gate_input);
      const float weight = norm_weight_is_bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(norm_weight)[value])
                                               : static_cast<const float*>(norm_weight)[value];
      const int64_t out_offset = (static_cast<int64_t>(token) * HV + value_head) * kDimV + value;
      const __nv_bfloat16 y = __float2bfloat16(output_values[i] * rstd * weight * gate);
      if (out != nullptr) out[out_offset] = y;
      rounded[i] = __bfloat162float(y);
    }
    if (quant.q != nullptr) {
      float absmax = fmaxf(fmaxf(fabsf(rounded[0]), fabsf(rounded[1])), fmaxf(fabsf(rounded[2]), fabsf(rounded[3])));
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) absmax = fmaxf(absmax, __shfl_xor_sync(0xffffffffu, absmax, offset));
      const float group_scale = __fdiv_rn(fmaxf(absmax, kQuantEps), kFp8Max);
      const float qinv = __fdiv_rn(1.0f, group_scale);  // the reciprocal flavour (kernels/csrc/quant.cuh)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int value = lane + i * 32;
        const float qv = fminf(fmaxf(__fmul_rn(rounded[i], qinv), -kFp8Max), kFp8Max);
        quant.q[(static_cast<int64_t>(token) * HV + value_head) * kDimV + value] =
            __nv_cvt_float_to_fp8(qv, __NV_SATFINITE, __NV_E4M3);
      }
      if (lane == 0) quant.s[value_head * quant.s_ld + token] = group_scale;
    }
    // The records for the next fold: raw k (of the key head), v and a, b of this value head.
    const int64_t base = static_cast<int64_t>(token) * mixed_row;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      rec.k[rbase + t * 128 + dim] = mixed_qkv[base + H * kDimK + key_head * kDimK + dim];
      rec.v[rbase + t * 128 + dim] = mixed_qkv[base + 2 * H * kDimK + value_head * kDimV + dim];
    }
    if (lane == 0) {
      rec.ab[abase + t * 2] = a[static_cast<int64_t>(token) * a_row + value_head];
      rec.ab[abase + t * 2 + 1] = b[static_cast<int64_t>(token) * b_row + value_head];
    }
  }
}

// FOLD and MTP decode checkpoints. Grid: (rows, layers, HV).
template <typename StateT>
__global__ __launch_bounds__(kThreads, 2) void gdn_fold_kernel(
    StateT* __restrict__ state, int64_t state_slot_stride, int64_t state_layer_stride, const __nv_bfloat16* rk,
    const __nv_bfloat16* rv, const __nv_bfloat16* rab, int64_t rec_req_stride, int64_t rec_layer_stride, int width,
    const float* __restrict__ a_log, const void* __restrict__ dt_bias, int dt_bias_type,
    const int64_t* __restrict__ req, const int64_t* __restrict__ src, const int64_t* __restrict__ dst,
    const int64_t* __restrict__ count, int HV) {
  const int row = blockIdx.x, layer = blockIdx.y, value_head = blockIdx.z;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int n = static_cast<int>(count[row]);
  n = n < 0 ? 0 : (n > width ? width : n);
  const int64_t s0 = src[row], s1 = dst[row];
  if (s0 <= 0 || s1 <= 0 || (n == 0 && s0 == s1)) return;
  __shared__ StateT shared_state[2][kChunkV][kDimK];
  __shared__ TokenSmem sm;
  const int64_t head_off = static_cast<int64_t>(layer) * state_layer_stride + value_head * kDimV * kDimK;
  const StateT* from = state + s0 * state_slot_stride + head_off;
  StateT* to = state + s1 * state_slot_stride + head_off;
  copy_state_chunk<StateT>(&shared_state[0][0][0], from, 0, tid);
  const int64_t rbase = req[row] * rec_req_stride + layer * rec_layer_stride + static_cast<int64_t>(value_head) * width * 128;
  const int64_t abase = req[row] * (rec_req_stride / 64) + layer * (rec_layer_stride / 64) +
                        static_cast<int64_t>(value_head) * width * 2;
  if (warp < n) {
    const int t = warp;
    load_token(sm, t, lane, rk + rbase + t * 128, rv + rbase + t * 128, nullptr, 1.0f,
               __bfloat162float(rab[abase + t * 2]), __bfloat162float(rab[abase + t * 2 + 1]),
               a_log[layer * HV + value_head], load_dt_bias(dt_bias, layer * HV + value_head, dt_bias_type), 0);
  }
  __syncthreads();
  const int k_base = lane * 4;
  int rows[kRowsPerWarp];
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) rows[r] = warp + r * kWarps;
#pragma unroll
  for (int chunk = 0; chunk < kNumChunks; ++chunk) {
    cp_async_wait_all();
    __syncthreads();
    if (chunk + 1 < kNumChunks) copy_state_chunk<StateT>(&shared_state[0][0][0], from, chunk + 1, tid);
    float h[kRowsPerWarp][4];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      const float4 sv = load_state4(&shared_state[chunk & 1][rows[r]][k_base]);
      h[r][0] = sv.x;
      h[r][1] = sv.y;
      h[r][2] = sv.z;
      h[r][3] = sv.w;
    }
    for (int t = 0; t < n; ++t) update(h, sm, t, k_base, chunk, rows, lane, false, 0, nullptr);
    store_rows(to, h, chunk, rows, k_base);
  }
}

template <typename StateT, bool SigmoidGate>
int launch_mtp(int ratio, dim3 grid, const void* mixed_qkv, int64_t mixed_row, const void* a, int64_t a_row,
               const void* b, int64_t b_row, const void* a_log, const void* dt_bias, int dt_bias_type,
               const void* state_slots, const void* req_slots, const void* fold_n, void* state, int64_t state_stride,
               const void* gate, int64_t gate_row, const void* norm_weight, bool norm_bf16, void* out, Quant quant,
               Records rec, int H, int HV, float scale, float eps, cudaStream_t stream) {
#define LMK_MTP_CASE(R)                                                                                              \
  case R:                                                                                                            \
    return lmk_launch(gdn_mtp_kernel<StateT, R, SigmoidGate>, grid, dim3(kThreads), 0, stream,                       \
                      (const __nv_bfloat16*)mixed_qkv, mixed_row, (const __nv_bfloat16*)a, a_row,                    \
                      (const __nv_bfloat16*)b, b_row, (const float*)a_log, dt_bias, dt_bias_type,                    \
                      (const int*)state_slots, (const int*)req_slots, (const int*)fold_n, (StateT*)state,            \
                      state_stride, (const __nv_bfloat16*)gate, gate_row, norm_weight, norm_bf16,                    \
                      (__nv_bfloat16*)out, quant, rec, H, HV, scale, eps);
  switch (ratio) {
    LMK_MTP_CASE(1)
    LMK_MTP_CASE(2)
    LMK_MTP_CASE(3)
    LMK_MTP_CASE(4)
    LMK_MTP_CASE(8)
    default:
      return -1;
  }
#undef LMK_MTP_CASE
}

}  // namespace

// MTP verify rows (kernels/ops.py `gdn_decode_mtp`). mixed_qkv [N * width, 2 H 128 + HV 128] (row stride);
// a, b [N * width, HV]; output_gate [N * width, HV, 128] (row stride, contiguous heads); state_slots, req_slots int32
// [N]; fold_n int32 [req slots]; state [slots, HV, 128, 128] at slot stride `state_stride`; records rec_k, rec_v [req
// slots, HV, width, 128] and rec_ab [req slots, HV, width, 2] at request stride rec_stride (rec_ab: rec_stride / 64);
// out [N * width, HV, 128] or null; q_out, s_out: the optional FP8 output (gdn_decode.cu).
LMK_API int lmk_gdn_mtp(const void* mixed_qkv, int64_t mixed_stride, const void* a, int64_t a_stride, const void* b,
                        int64_t b_stride, const void* a_log, const void* dt_bias, int dt_bias_type,
                        const void* state_slots, const void* req_slots, const void* fold_n, void* state,
                        int64_t state_stride, int state_is_bf16, const void* output_gate, int64_t gate_stride,
                        const void* norm_weight, int norm_weight_is_bf16, void* out, void* q_out, void* s_out,
                        int64_t s_ld, void* rec_k, void* rec_v, void* rec_ab, int64_t rec_stride, int width,
                        int num_rows, int H, int HV, float scale, float norm_eps, int sigmoid_gate,
                        cudaStream_t stream) {
  if (num_rows <= 0) return 0;
  if (HV % H || width < 1 || width > kMaxWin || (out == nullptr && q_out == nullptr)) return -1;
  const Quant quant{static_cast<uint8_t*>(q_out), static_cast<float*>(s_out), s_ld};
  const Records rec{static_cast<__nv_bfloat16*>(rec_k), static_cast<__nv_bfloat16*>(rec_v),
                    static_cast<__nv_bfloat16*>(rec_ab), rec_stride, width};
  const dim3 grid(num_rows, HV);
  const int ratio = HV / H;
#define LMK_MTP_ARGS                                                                                                 \
  ratio, grid, mixed_qkv, mixed_stride, a, a_stride, b, b_stride, a_log, dt_bias, dt_bias_type, state_slots,         \
      req_slots, fold_n, state, state_stride, output_gate, gate_stride, norm_weight, norm_weight_is_bf16 != 0, out,  \
      quant, rec, H, HV, scale, norm_eps, stream
  if (state_is_bf16) {
    return sigmoid_gate ? launch_mtp<__nv_bfloat16, true>(LMK_MTP_ARGS) : launch_mtp<__nv_bfloat16, false>(LMK_MTP_ARGS);
  }
  return sigmoid_gate ? launch_mtp<float, true>(LMK_MTP_ARGS) : launch_mtp<float, false>(LMK_MTP_ARGS);
#undef LMK_MTP_ARGS
}

// FOLD and MTP decode checkpoints for every GDN layer (kernels/ops.py `gdn_fold`): state [slots, layers, HV, 128, 128]
// contiguous; rec_k, rec_v [req slots, layers, HV, width, 128], rec_ab [req slots, layers, HV, width, 2] contiguous;
// a_log [layers, HV] fp32, dt_bias [layers, HV]; req, src, dst, n int64 [rows]. A row with src <= 0 or dst <= 0, or
// with n == 0 and src == dst, does nothing.
LMK_API int lmk_gdn_fold(void* state, const void* rec_k, const void* rec_v, const void* rec_ab, const void* a_log,
                         const void* dt_bias, int dt_bias_type, const void* req, const void* src, const void* dst,
                         const void* n, int rows, int layers, int HV, int width, int state_is_bf16,
                         cudaStream_t stream) {
  if (rows <= 0) return 0;
  if (width < 1 || width > kMaxWin) return -1;
  const int64_t layer_stride = static_cast<int64_t>(HV) * kDimV * kDimK;
  const int64_t slot_stride = layer_stride * layers;
  const int64_t rec_layer = static_cast<int64_t>(HV) * width * 128;
  const int64_t rec_req = rec_layer * layers;
  const dim3 grid(rows, layers, HV);
  if (state_is_bf16) {
    gdn_fold_kernel<__nv_bfloat16><<<grid, kThreads, 0, stream>>>(
        (__nv_bfloat16*)state, slot_stride, layer_stride, (const __nv_bfloat16*)rec_k, (const __nv_bfloat16*)rec_v,
        (const __nv_bfloat16*)rec_ab, rec_req, rec_layer, width, (const float*)a_log, dt_bias, dt_bias_type,
        (const int64_t*)req, (const int64_t*)src, (const int64_t*)dst, (const int64_t*)n, HV);
  } else {
    gdn_fold_kernel<float><<<grid, kThreads, 0, stream>>>(
        (float*)state, slot_stride, layer_stride, (const __nv_bfloat16*)rec_k, (const __nv_bfloat16*)rec_v,
        (const __nv_bfloat16*)rec_ab, rec_req, rec_layer, width, (const float*)a_log, dt_bias, dt_bias_type,
        (const int64_t*)req, (const int64_t*)src, (const int64_t*)dst, (const int64_t*)n, HV);
  }
  return (int)cudaGetLastError();
}
