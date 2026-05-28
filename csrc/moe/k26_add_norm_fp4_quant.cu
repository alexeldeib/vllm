#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <tuple>

#include "core/registration.h"
#include "dsv3_router_gemm_utils.h"

namespace {

constexpr int kHidden = 7168;
constexpr int kPack = 16;
constexpr int kCols = kHidden / kPack;
constexpr int kScaleRows = 128;
constexpr int kScaleColsI32 = (kHidden / kPack) / 4;
constexpr int kWarps = kCols / 32;

__device__ __forceinline__ float rcp_approx_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
  return b;
}

__device__ __forceinline__ uint8_t fp8_e4m3(float x) {
  __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(x);
  uint8_t out;
  reinterpret_cast<__nv_fp8_e4m3&>(out) = tmp;
  return out;
}

__device__ __forceinline__ uint32_t fp32_vec8_to_e2m1(float (&array)[8]) {
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]),
        "f"(array[4]), "f"(array[5]), "f"(array[6]), "f"(array[7]));
  return val;
}

__device__ __forceinline__ void pack_fp4x16(float (&vals)[16], uint32_t& lo,
                                            uint32_t& hi) {
  float lo_vals[8];
  float hi_vals[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    lo_vals[i] = vals[i];
    hi_vals[i] = vals[i + 8];
  }
  lo = fp32_vec8_to_e2m1(lo_vals);
  hi = fp32_vec8_to_e2m1(hi_vals);
}

__device__ __forceinline__ uint8_t* swizzled_sf_ptr(int row, int col,
                                                    int32_t* scale_i32) {
  int32_t const kTileIdx = col >> 2;
  int32_t const innerKIdx = col & 3;
  int32_t const outerMIdx = row & 31;
  int32_t const innerMIdx = (row >> 5) & 3;
  int64_t const offset =
      (static_cast<int64_t>(kTileIdx) << 9) | (outerMIdx << 4) |
      (innerMIdx << 2) | innerKIdx;
  return reinterpret_cast<uint8_t*>(scale_i32) + offset;
}

__device__ __forceinline__ float warp_sum(float v) {
  v += __shfl_xor_sync(0xffffffff, v, 16);
  v += __shfl_xor_sync(0xffffffff, v, 8);
  v += __shfl_xor_sync(0xffffffff, v, 4);
  v += __shfl_xor_sync(0xffffffff, v, 2);
  v += __shfl_xor_sync(0xffffffff, v, 1);
  return v;
}

__global__ __launch_bounds__(kCols, 1) void add_norm_fp4_quant_kernel(
    uint8_t* __restrict__ out_fp4, int32_t* __restrict__ out_scale_i32,
    __nv_bfloat16 const* __restrict__ x,
    __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16 const* __restrict__ norm_weight,
    float const* __restrict__ input_global_scale, int num_tokens, float eps) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  int const row = blockIdx.x;
  int const col = threadIdx.x;
  if (row >= num_tokens || col >= kCols) return;

  int const lane = threadIdx.x & 31;
  int const warp = threadIdx.x >> 5;
  int const base = row * kHidden + col * kPack;

  float a_vals[kPack];
  float ss = 0.0f;

#pragma unroll
  for (int i = 0; i < kPack; i++) {
    float a =
        __bfloat162float(x[base + i]) + __bfloat162float(residual[base + i]);
    __nv_bfloat16 a_bf16 = __float2bfloat16(a);
    residual[base + i] = a_bf16;
    float ar = __bfloat162float(a_bf16);
    a_vals[i] = ar;
    ss += ar * ar;
  }

  __shared__ float warp_ss[kWarps];
  __shared__ float s_rsqrt;

  float wsum = warp_sum(ss);
  if (lane == 0) {
    warp_ss[warp] = wsum;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; w++) {
      total += warp_ss[w];
    }
    s_rsqrt = rsqrtf(total / static_cast<float>(kHidden) + eps);
  }
  __syncthreads();

  float normed[kPack];
  float local_abs_max = 0.0f;
  float const rs = s_rsqrt;
#pragma unroll
  for (int i = 0; i < kPack; i++) {
    float n = a_vals[i] * rs * __bfloat162float(norm_weight[col * kPack + i]);
    n = __bfloat162float(__float2bfloat16(n));
    normed[i] = n;
    local_abs_max = fmaxf(local_abs_max, fabsf(n));
  }

  float const sf_scale =
      input_global_scale == nullptr ? 1.0f : input_global_scale[0];
  float sf_value = sf_scale * (local_abs_max * rcp_approx_ftz(6.0f));
  uint8_t sf8 = fp8_e4m3(sf_value);
  *swizzled_sf_ptr(row, col, out_scale_i32) = sf8;

  __nv_fp8_e4m3 sf8_val;
  reinterpret_cast<uint8_t&>(sf8_val) = sf8;
  sf_value = float(sf8_val);
  float output_scale =
      sf_value != 0.0f ? rcp_approx_ftz(sf_value * rcp_approx_ftz(sf_scale))
                       : 0.0f;

  float q_vals[kPack];
#pragma unroll
  for (int i = 0; i < kPack; i++) {
    q_vals[i] = normed[i] * output_scale;
  }

  uint32_t lo, hi;
  pack_fp4x16(q_vals, lo, hi);
  int64_t out_u64_idx =
      (static_cast<int64_t>(row) * (kHidden / 8) + col * 2) >> 1;
  reinterpret_cast<uint64_t*>(out_fp4)[out_u64_idx] =
      (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
#endif
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> k26_add_norm_fp4_quant(
    at::Tensor const& x, at::Tensor& residual, at::Tensor const& norm_weight,
    at::Tensor const& input_global_scale, double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(residual.is_cuda(), "residual must be CUDA");
  TORCH_CHECK(norm_weight.is_cuda(), "norm_weight must be CUDA");
  TORCH_CHECK(input_global_scale.is_cuda(), "input_global_scale must be CUDA");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bf16");
  TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be bf16");
  TORCH_CHECK(norm_weight.scalar_type() == at::kBFloat16,
              "norm_weight must be bf16");
  TORCH_CHECK(input_global_scale.scalar_type() == at::kFloat,
              "input_global_scale must be fp32");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  TORCH_CHECK(norm_weight.is_contiguous(), "norm_weight must be contiguous");
  TORCH_CHECK(input_global_scale.is_contiguous(),
              "input_global_scale must be contiguous");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == kHidden, "x must be [M,7168]");
  TORCH_CHECK(residual.sizes() == x.sizes(), "residual shape mismatch");
  TORCH_CHECK(norm_weight.numel() == kHidden, "norm_weight shape mismatch");
  TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 16, "M must be in [1,16]");

  static int const sm = getSMVersion();
  TORCH_CHECK(sm >= 100 && sm <= 103,
              "k26_add_norm_fp4_quant requires SM_100 <= CUDA ARCH <= SM_103");

  auto out_fp4 =
      at::empty({x.size(0), kHidden / 2}, x.options().dtype(at::kByte));
  auto out_scale_i32 =
      at::empty({kScaleRows, kScaleColsI32}, x.options().dtype(at::kInt));

  cudaStream_t const stream = at::cuda::getCurrentCUDAStream();
  cudaMemsetAsync(out_scale_i32.mutable_data_ptr(), 0,
                  out_scale_i32.numel() * sizeof(int32_t), stream);

  add_norm_fp4_quant_kernel<<<static_cast<int>(x.size(0)), kCols, 0, stream>>>(
      reinterpret_cast<uint8_t*>(out_fp4.mutable_data_ptr()),
      reinterpret_cast<int32_t*>(out_scale_i32.mutable_data_ptr()),
      reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(residual.mutable_data_ptr()),
      reinterpret_cast<__nv_bfloat16 const*>(norm_weight.data_ptr()),
      reinterpret_cast<float const*>(input_global_scale.data_ptr()),
      static_cast<int>(x.size(0)), static_cast<float>(eps));

  return {out_fp4, out_scale_i32};
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("k26_add_norm_fp4_quant", &k26_add_norm_fp4_quant);
}
