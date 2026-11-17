// Shared helpers for llm-mini-infer's CUDA kernels (docs/architecture.md, "Kernels"). Plain CUDA, no torch headers.
// nvcc alone builds the library (llm_mini_infer/kernels/build.py). Callers use ctypes with raw device pointers and
// their own stream.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define LMK_API extern "C" __attribute__((visibility("default")))

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float bf2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2bf(float x) { return __float2bfloat16_rn(x); }

// Rounded fp32 operations. The torch reference runs one op per kernel, so it contracts nothing into an FMA. Kernels
// that must reproduce its rounding use these instead of `*`, `+`, `-`.
__device__ __forceinline__ float fmul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float fadd(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float fsub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float fdiv(float a, float b) { return __fdiv_rn(a, b); }

// torch's CUDA formulas (each op rounded to fp32): sigmoid = 1 / (1 + exp(-x)), silu = x / (1 + exp(-x)).
__device__ __forceinline__ float t_sigmoid(float x) { return fdiv(1.0f, fadd(1.0f, expf(-x))); }
__device__ __forceinline__ float t_silu(float x) { return fdiv(x, fadd(1.0f, expf(-x))); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Sum over the block (blockDim.x a multiple of 32, at most 1024). Every thread gets the result. `smem` holds 32 floats.
__device__ __forceinline__ float block_sum(float v, float* smem) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = warp_sum(v);
  __syncthreads();  // a previous call may still read smem
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  v = lane < nw ? smem[lane] : 0.0f;
  return warp_sum(v);
}

static inline int lmk_status() { return (int)cudaGetLastError(); }

// Programmatic dependent launch (PDL; docs/architecture.md, "CUDA graphs"). While the library's flag is on
// (`lmk_set_pdl`, csrc/launch.cu), `lmk_launch` adds the programmatic-stream-serialization attribute. The kernel may
// then start while the previous kernel on the stream finishes. In a CUDA graph, the edge between them becomes a
// programmatic one.
// Every kernel launched that way calls `lmk_pdl_wait()` in every CTA before it reads or writes memory that a
// predecessor may touch. Only constant inputs (the weights) may be read before it. The wait returns when the previous
// kernel has completed and its writes are visible, so every bit equals the serialized launch's. Without the attribute
// (flag off, the default), the wait returns at once.
// `lmk_pdl_trigger()` lets the next kernel launch before this one ends. That kernel then waits in its own
// `lmk_pdl_wait`. Without a trigger, the next kernel launches as this one's CTAs exit.
// The trigger lets only a successor that has the attribute start early. Our kernels and the CUTLASS GEMMs carry the
// attribute only while the flag is on. FlashInfer's attention carries it by default (SM 9.0+). So the kernels before
// FlashInfer (the attention inputs: Gemma 4's `g4_qkv` and Qwen's `qk_norm_rope`) trigger only while the flag is on
// (an instantiation each way, `kTrigger`). With the flag off, FlashInfer starts at their end, as after a serialized
// launch.
__device__ __forceinline__ void lmk_pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void lmk_pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// The library's PDL flag: one per process (the static of an inline function is shared by every translation unit).
inline int& lmk_pdl_flag() {
  static int on = 0;
  return on;
}

// kernel<<<grid, block, smem, stream>>>(args...), with the PDL attribute while the flag is on. Returns a cudaError_t.
template <typename... Params, typename... Args>
static inline int lmk_launch(void (*kernel)(Params...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                             Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  if (lmk_pdl_flag()) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
  }
  return (int)cudaLaunchKernelEx(&cfg, kernel, static_cast<Params>(args)...);
}
