// Shared device helpers for the FP8 activation quantization (quant.cu, and the quantizing producers in gemma.cu).
// The numerics, vLLM's: scale = max(absmax, 1e-10) / 448, an IEEE division; then q = e4m3_rn(clamp(x * inv, -448, 448))
// with inv = 1 / scale, one IEEE division a group and a product rounded to nearest, as vLLM's standalone and fused-norm
// group quantizers compute it. The SiLU-and-mul fusion divides instead, q = e4m3_rn(clamp(x / scale, -448, 448)), as
// vLLM's fused silu_and_mul_per_block_quant does (`kDivide`). The reference is `quant_ref` in kernels/fp8.py.
#pragma once

#include <cuda_fp8.h>

#include "common.cuh"

namespace lmk_quant {

constexpr float kFp8Max = 448.0f;
constexpr float kQuantEps = 1e-10f;
constexpr int kGroup = 128;

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ float group_scale(float absmax) { return fdiv(fmaxf(absmax, kQuantEps), kFp8Max); }

// What a group's values are multiplied by: 1 / scale. An Inf scale gives 0, the floor's (1e-10 / 448) about 4.5e12.
__device__ __forceinline__ float group_inv(float scale) { return fdiv(1.0f, scale); }

__device__ __forceinline__ uint32_t e4m3(float y) {
  const float q = fminf(fmaxf(y, -kFp8Max), kFp8Max);  // a NaN gives -448
  return static_cast<uint32_t>(__nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3));
}

// One byte of x * inv (`group_inv`): every producer but SiLU-and-mul.
__device__ __forceinline__ uint32_t fp8_byte(float x, float inv) { return e4m3(fmul(x, inv)); }

// One byte of x / scale: the SiLU-and-mul fusion's form.
__device__ __forceinline__ uint32_t fp8_byte_div(float x, float scale) { return e4m3(fdiv(x, scale)); }

// One warp quantizes one (token, group). Lane l holds values 4l to 4l + 3 of the group and writes the 4 bytes at
// q_group + 4l. Lane 0 writes the scale. kDivide: the SiLU-and-mul fusion's division.
template <bool kDivide = false>
__device__ __forceinline__ void quant_store4(const float (&v)[4], uint8_t* q_group, float* scale_out, int lane) {
  float am = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
  am = warp_max(am);
  const float sc = group_scale(am);
  uint32_t packed;
  if constexpr (kDivide) {
    packed = fp8_byte_div(v[0], sc) | (fp8_byte_div(v[1], sc) << 8) | (fp8_byte_div(v[2], sc) << 16) |
             (fp8_byte_div(v[3], sc) << 24);
  } else {
    const float inv = group_inv(sc);
    packed = fp8_byte(v[0], inv) | (fp8_byte(v[1], inv) << 8) | (fp8_byte(v[2], inv) << 16) |
             (fp8_byte(v[3], inv) << 24);
  }
  *reinterpret_cast<uint32_t*>(q_group + 4 * lane) = packed;
  if (lane == 0) *scale_out = sc;
}

__device__ __forceinline__ void load_bf16x4(const bf16* p, float (&v)[4]) {
  const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(p);
  const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(p + 2);
  v[0] = __low2float(lo);
  v[1] = __high2float(lo);
  v[2] = __low2float(hi);
  v[3] = __high2float(hi);
}

}  // namespace lmk_quant
