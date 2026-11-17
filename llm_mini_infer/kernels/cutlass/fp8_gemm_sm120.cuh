/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * The block-FP8 GEMM for SM120/121 (docs/architecture.md, "FP8"). It is vLLM 0.30's CUTLASS blockwise kernel
 * (`csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8{,_dispatch.cuh}`, Apache-2.0),
 * vendored. Changes for llm-mini-infer: plain CUDA behind an extern "C" launcher (raw pointers, the caller's stream and
 * a caller-owned workspace, so a CUDA graph can capture it; no torch headers); the caller chooses the configuration
 * (kernels/fp8.py dispatches by M from measurements); the caller passes the SM count instead of a query per call;
 * bf16 output only. The kernels and their tile shapes are vLLM's.
 *
 *   D[m, n] = sum_kb (sum_{k in kb} A[m, k] B[n, k]) * SFA[kb, m] * SFB[n / 128, kb]      (bf16 out, fp32 accumulate)
 *
 * A [m, k] e4m3 row-major (the activations). B [n, k] e4m3 row-major (the weight as stored: column-major K x N).
 * SFA [k / 128, m] fp32 (MN-major: one scale per token and 128-wide K group). SFB [n / 128, k / 128] fp32 (the
 * checkpoint's weight_scale_inv layout). D [m, n] bf16 row-major. n and k must be multiples of 128.
 *
 * Each configuration has its own translation unit (fp8_gemm_sm120_c{0,1,2,3}.cu, compiled in parallel).
 * fp8_gemm_sm120.cu holds the extern "C" launchers.
 *
 * Configurations (the original kernel's dispatch: M <= 64 swap-AB, M <= 256 ping-pong, else cooperative):
 *   0: swap-AB, 128x32x128 tiles, cooperative (the weight is the MMA's A operand: 128 rows of N per tile);
 *   1: 64x128x128 tiles, ping-pong;
 *   2: 128x128x128 tiles, cooperative;
 *   3: swap-AB, 128x64x128 tiles, cooperative: SGLang's SM120 tile for 128 < M <= 256 (sglang/kernels/jit/csrc/gemm/
 *      fp8_blockwise/fp8_blockwise_scaled_mm_sm120.cuh, autotuned on Qwen3.x-27B's shapes).
 */
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

#define LMK_API extern "C" __attribute__((visibility("default")))

namespace lmk_fp8 {

using namespace cute;

// vLLM's enable_sm120_family (cutlass_extensions/common.hpp): the kernel body only for SM 12.x.
template <typename Kernel>
struct enable_sm120_family : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__
#if (__CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300)
    Kernel::operator()(std::forward<Args>(args)...);
#else
    printf("This kernel only supports sm120f.\n");
    asm("trap;");
#endif
#endif
  }
};

// clang-format off
template <class OutType, int ScaleGranularityM, int ScaleGranularityN, int ScaleGranularityK, class MmaTileShape,
          class ClusterShape, class EpilogueScheduler, class MainloopScheduler, bool swap_ab_ = false>
struct cutlass_3x_gemm_fp8_blockwise {
  static constexpr bool swap_ab = swap_ab_;
  using ElementAB = cutlass::float_e4m3_t;

  using ElementA = ElementAB;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = ElementAB;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementD = OutType;
  using LayoutD = cutlass::layout::RowMajor;
  using LayoutD_Transpose = typename cutlass::layout::LayoutTranspose<LayoutD>::type;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementC = void;
  using LayoutC = LayoutD;
  using LayoutC_Transpose = LayoutD_Transpose;
  static constexpr int AlignmentC = AlignmentD;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementBlockScale = float;

  using ScaleConfig = conditional_t<swap_ab,
      cutlass::detail::Sm120BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
                                                 cute::UMMA::Major::K, cute::UMMA::Major::MN>,
      cutlass::detail::Sm120BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
                                                 cute::UMMA::Major::MN, cute::UMMA::Major::K>>;

  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  using ElementScalar = float;
  using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC,
                                                                         ElementScalar, RoundStyle>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, MmaTileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute, ElementC, conditional_t<swap_ab, LayoutC_Transpose, LayoutC>, AlignmentC,
      ElementD, conditional_t<swap_ab, LayoutD_Transpose, LayoutD>, AlignmentD, EpilogueScheduler,
      DefaultOperation>::CollectiveOp;

  using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;
  using CollectiveMainloop = conditional_t<swap_ab,
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementB, cute::tuple<LayoutB_Transpose, LayoutSFA>, AlignmentB, ElementA,
          cute::tuple<LayoutA_Transpose, LayoutSFB>, AlignmentA, ElementAccumulator, MmaTileShape, ClusterShape,
          StageCount, MainloopScheduler>::CollectiveOp,
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA, ElementB,
          cute::tuple<LayoutB, LayoutSFB>, AlignmentB, ElementAccumulator, MmaTileShape, ClusterShape, StageCount,
          MainloopScheduler>::CollectiveOp>;

  using KernelType = enable_sm120_family<cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>>;

  struct GemmKernel : public KernelType {};
};

// The vendored kernel's three configurations (0 to 2); config 3 below is SGLang's tile.
template <typename OutType>
struct config_cooperative {  // 128x128x128, cooperative (the builder's Auto choice)
  using Gemm = cutlass_3x_gemm_fp8_blockwise<
      OutType, 1, 128, 128, Shape<_128, _128, _128>, Shape<_1, _1, _1>,
      cutlass::epilogue::collective::EpilogueScheduleAuto, cutlass::gemm::collective::KernelScheduleAuto>;
};

template <typename OutType>
struct config_pingpong {  // 64x128x128, ping-pong
  using Gemm = cutlass_3x_gemm_fp8_blockwise<
      OutType, 1, 128, 128, Shape<_64, _128, _128>, Shape<_1, _1, _1>,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120>;
};

template <typename OutType>
struct config_swapab {  // swap A/B: 128 (of N) x 32 (of M) x 128, cooperative
  using Gemm = cutlass_3x_gemm_fp8_blockwise<
      OutType, 128, 1, 128, Shape<_128, _32, _128>, Shape<_1, _1, _1>,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120, true>;
};
template <typename OutType>
struct config_swapab64 {  // swap A/B: 128 (of N) x 64 (of M) x 128, cooperative (SGLang's tile for M in (128, 256])
  using Gemm = cutlass_3x_gemm_fp8_blockwise<
      OutType, 128, 1, 128, Shape<_128, _64, _128>, Shape<_1, _1, _1>,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120, true>;
};
// clang-format on

template <typename Gemm>
typename Gemm::GemmKernel::Arguments make_args(const void* a, const void* b, const void* sa, const void* sb, void* d,
                                               int m, int n, int k, int swizzle, int sm_count, int device) {
  static constexpr bool swap_ab = Gemm::swap_ab;
  using GemmKernel = typename Gemm::GemmKernel;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using ScaleConfig = typename Gemm::ScaleConfig;
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;

  StrideA a_stride = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  StrideB b_stride = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  StrideC c_stride = cutlass::make_cute_packed_stride(
      StrideC{}, swap_ab ? cute::make_shape(n, m, 1) : cute::make_shape(m, n, 1));
  auto layout_SFA = swap_ab ? ScaleConfig::tile_atom_to_shape_SFA(make_shape(n, m, k, 1))
                            : ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  auto layout_SFB = swap_ab ? ScaleConfig::tile_atom_to_shape_SFB(make_shape(n, m, k, 1))
                            : ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

  auto a_ptr = static_cast<ElementAB const*>(a);
  auto b_ptr = static_cast<ElementAB const*>(b);
  auto sa_ptr = static_cast<float const*>(sa);
  auto sb_ptr = static_cast<float const*>(sb);

  typename GemmKernel::MainloopArguments mainloop{};
  mainloop.layout_SFA = layout_SFA;
  mainloop.layout_SFB = layout_SFB;
  if (swap_ab) {
    mainloop.ptr_A = b_ptr;
    mainloop.dA = b_stride;
    mainloop.ptr_B = a_ptr;
    mainloop.dB = a_stride;
    mainloop.ptr_SFA = sb_ptr;
    mainloop.ptr_SFB = sa_ptr;
  } else {
    mainloop.ptr_A = a_ptr;
    mainloop.dA = a_stride;
    mainloop.ptr_B = b_ptr;
    mainloop.dB = b_stride;
    mainloop.ptr_SFA = sa_ptr;
    mainloop.ptr_SFB = sb_ptr;
  }
  auto d_ptr = static_cast<ElementD*>(d);
  typename GemmKernel::EpilogueArguments epilogue{{}, d_ptr, c_stride, d_ptr, c_stride};
  typename GemmKernel::TileSchedulerArguments scheduler{};
  scheduler.max_swizzle_size = swizzle;  // 8 when the weight exceeds the L2, else 1 (bit-identical either way)
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device;
  hw_info.sm_count = sm_count;
  auto shape = swap_ab ? cute::make_shape(n, m, k, 1) : cute::make_shape(m, n, k, 1);
  return typename GemmKernel::Arguments{cutlass::gemm::GemmUniversalMode::kGemm, shape, mainloop, epilogue, hw_info,
                                        scheduler};
}

template <typename Gemm>
int64_t workspace_bytes(int m, int n, int k, int swizzle, int sm_count, int device) {
  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<typename Gemm::GemmKernel>;
  auto args = make_args<Gemm>(nullptr, nullptr, nullptr, nullptr, nullptr, m, n, k, swizzle, sm_count, device);
  return static_cast<int64_t>(GemmOp::get_workspace_size(args));
}

// `pdl`: launch with programmatic dependent launch (kernels/csrc/common.cuh). The library builds with
// CUTLASS_ENABLE_GDC_FOR_SM100 (kernels/build.py), so every producer warp waits for the previous kernel before its
// first load (CUTLASS's sm90 warp-specialized kernels, which the SM120 blockwise kernels use). Without the attribute
// the wait returns at once, so the bits are the same either way.
template <typename Gemm>
int run(const void* a, const void* b, const void* sa, const void* sb, void* d, int m, int n, int k, int swizzle,
        int sm_count, int device, void* workspace, int64_t workspace_bytes, bool pdl, cudaStream_t stream) {
  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<typename Gemm::GemmKernel>;
  auto args = make_args<Gemm>(a, b, sa, sb, d, m, n, k, swizzle, sm_count, device);
  GemmOp op;
  if (op.can_implement(args) != cutlass::Status::kSuccess) return -1;
  if (static_cast<int64_t>(GemmOp::get_workspace_size(args)) > workspace_bytes) return -2;
  if (op.run(args, workspace, stream, nullptr, pdl) != cutlass::Status::kSuccess) return -3;
  return static_cast<int>(cudaGetLastError());
}

// The entry points of one configuration, defined in its own translation unit.
#define LMK_FP8_CONFIG_DECL(N)                                                                                     \
  int run_c##N(const void* a, const void* b, const void* sa, const void* sb, void* d, int m, int n, int k,       \
               int swizzle, int sm_count, int device, void* workspace, int64_t workspace_bytes, bool pdl,        \
               cudaStream_t s);                                                                                  \
  int64_t workspace_c##N(int m, int n, int k, int swizzle, int sm_count, int device);
LMK_FP8_CONFIG_DECL(0)
LMK_FP8_CONFIG_DECL(1)
LMK_FP8_CONFIG_DECL(2)
LMK_FP8_CONFIG_DECL(3)

}  // namespace lmk_fp8
