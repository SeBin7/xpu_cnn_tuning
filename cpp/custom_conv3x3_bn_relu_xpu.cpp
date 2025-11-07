// custom_conv3x3_bn_relu_xpu.cpp (merged: GLOBAL + IM2ROW + SLM + Winograd)
// Fused 3x3 Conv + (optional BN fold via bias) + ReLU for XPU (SYCL)
// Paths:
//   backend=aten    : delegate to at::conv2d (+bias) + ReLU (leverages IPEX/oneDNN on XPU)
//   layout=direct   : custom GLOBAL vec4 + KO_STEP
//   layout=im2row   : row-pack microkernel (lightweight, also used for sgemm mode)
//   layout=slm      : halo tile in local memory (stride=1 only)
//   algo=winograd   : F(2x2,3x3) (stride=1, pad=1)
//   algo=sgemm      : map to im2row microkernel (no external BLAS dependency)
// All comments in English only.

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/conv2d.h>
#include <ATen/ops/relu.h>
#include <torch/extension.h>

#include <sycl/sycl.hpp>
#if __has_include(<ATen/xpu/XPUStream.h>)
  #include <ATen/xpu/XPUStream.h>
  using at::xpu::getCurrentXPUStream;
#elif __has_include(<c10/xpu/XPUStream.h>)
  #include <c10/xpu/XPUStream.h>
  using c10::xpu::getCurrentXPUStream;
#else
  #error "XPUStream header not found."
#endif

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <tuple>
#include <string>
#include <algorithm>
#include <memory>
#include <iostream>
#include <utility>

namespace s = sycl;
using at::Tensor;

// ======================= Env helpers =======================
static inline int get_env_i(const char* key, int defv) {
  if (const char* v = std::getenv(key)) return std::max(1, std::atoi(v));
  return defv;
}
static inline bool get_env_bool_01(const char* key, bool defv) {
  if (const char* v = std::getenv(key)) return std::atoi(v) != 0;
  return defv;
}
static inline std::pair<int,int> parse_tile_env(const char* key, int def_h, int def_w) {
  if (const char* v = std::getenv(key)) { int H=0,W=0; if (std::sscanf(v, "%dx%d", &H, &W) == 2 && H>0 && W>0) return {H,W}; }
  return {def_h, def_w};
}
static inline std::string get_env_s(const char* key, const char* defv){
  if(const char* v = std::getenv(key)) return std::string(v);
  return std::string(defv);
}
static inline bool env_present(const char* key) {
  return std::getenv(key) != nullptr;
}
static inline int auto_ko_step(int64_t Ci) {
  // Optimize for better cache line utilization
  if (Ci >= 512) return 64;
  if (Ci >= 256) return 32;
  if (Ci >= 128) return 16;
  if (Ci >= 64)  return 16;  // Increased from 8 for better vectorization
  if (Ci >= 32)  return 8;   // Increased from 4
  if (Ci >= 16)  return 4;   // Increased from 2
  return 1;
}
static inline std::pair<int,int> auto_tile_hw(int64_t Ho, int64_t Wo, int def_h, int def_w) {
  auto pick = [](int64_t extent, int defv) -> int {
    if (extent <= 0) return 1;
    // Prefer larger tiles for better cache utilization
    if (extent >= 64) return std::min(defv, 64);
    if (extent >= 32) return std::min(defv, 32);
    if (extent >= 16) return std::min(defv, 16);
    if (extent >= 8)  return std::min(defv, 8);
    if (extent >= 4)  return std::min(defv, 4);
    if (extent >= 2)  return 2;
    return 1;
  };
  return {pick(Ho, def_h), pick(Wo, def_w)};
}

// ======================= Tunables (defaults) =======================
#ifndef TILE_H_S1_DEF
#define TILE_H_S1_DEF 8
#endif
#ifndef TILE_W_S1_DEF
#define TILE_W_S1_DEF 16
#endif
#ifndef TILE_H_S2_DEF
#define TILE_H_S2_DEF 4
#endif
#ifndef TILE_W_S2_DEF
#define TILE_W_S2_DEF 16
#endif
#ifndef USE_VEC4_S1_DEF
#define USE_VEC4_S1_DEF 1
#endif
#ifndef USE_VEC4_S2_DEF
#define USE_VEC4_S2_DEF 1
#endif

// ======================= Misc helpers =======================
static inline void check_input(const Tensor& t) {
  const bool is_nchw = t.is_contiguous();
  const bool is_cl = t.is_contiguous(at::MemoryFormat::ChannelsLast);
  TORCH_CHECK(is_nchw || is_cl, "Input must be contiguous NCHW or channels_last.");
  TORCH_CHECK(t.scalar_type() == at::kFloat, "Only float32 is supported.");
  TORCH_CHECK(t.dim() == 4, "Expected 4D tensor.");
}
static inline std::tuple<int64_t,int64_t> out_hw(
    int64_t H, int64_t W, int64_t k, int64_t s, int64_t p) {
  const int64_t Ho = (H + 2*p - k) / s + 1;
  const int64_t Wo = (W + 2*p - k) / s + 1;
  return {Ho, Wo};
}
static inline bool aligned_vec4_ptr(const float* p) { return (reinterpret_cast<uintptr_t>(p) % (4*sizeof(float))) == 0; }
static inline bool aligned_vec16_ptr(const float* p) { return (reinterpret_cast<uintptr_t>(p) % (16*sizeof(float))) == 0; }
static inline float dot3_v4_row(const s::vec<float,4>& v4, const float* w_row3) {
  return v4.x()*w_row3[0] + v4.y()*w_row3[1] + v4.z()*w_row3[2];
}
static inline float dot16(const s::vec<float,16>& a, const s::vec<float,16>& b) {
  float acc = 0.f;
  #pragma unroll
  for (int i=0; i<16; ++i) acc = fma(a[i], b[i], acc);
  return acc;
}
static inline float dot8(const s::vec<float,8>& a, const s::vec<float,8>& b) {
  float acc = 0.f;
  #pragma unroll
  for (int i=0; i<8; ++i) acc = fma(a[i], b[i], acc);
  return acc;
}
static inline bool aligned_vec8_ptr(const float* p) { return (reinterpret_cast<uintptr_t>(p) % (8*sizeof(float))) == 0; }
using atomic64 = sycl::atomic_ref<
    uint64_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>;
static inline float dot4_v4(const s::vec<float,4>& a, const s::vec<float,4>& b) {
  return a.x()*b.x() + a.y()*b.y() + a.z()*b.z() + a.w()*b.w();
}
static inline int clamp_idx(int v, int lo, int hi){ return v < lo ? lo : (v > hi ? hi : v); }

#define XIDX4(n,c,h,w,C,H,W) ((((int64_t)(n)*(C) + (c))*(H) + (h))*(W) + (w))
#define WIDX(co,ci,kh,kw,Ci) ((((int64_t)(co)*(Ci) + (ci))*3 + (kh))*3 + (kw))

// ============================================================================
// GLOBAL kernels (vec4 + KO_STEP)
// ============================================================================
static void launch_conv3x3_global_stride1(
    s::queue& q, bool use_v4, int ko_step,
    const float* __restrict__ x,   // [N,Ci,H,W]
    const float* __restrict__ w,   // [Co,Ci,3,3]
    const float* __restrict__ bias,// [Co] or nullptr
    float* __restrict__ y,         // [N,Co,Ho,Wo]
    int64_t N, int64_t Ci, int64_t H, int64_t W,
    int64_t Co, int64_t Ho, int64_t Wo,
    int pad, int TH, int TW)
{
  const int64_t HW = H * W;
  const int64_t CiHW = Ci * HW;
  const int64_t HoWo = Ho * Wo;
  const s::range<3> lrange(1, TH, TW);
  const s::range<3> grange(N*Co, (Ho+TH-1)/TH, (Wo+TW-1)/TW);
  const s::nd_range<3> ndr(grange * lrange, lrange);

  q.submit([&](s::handler& h) {
    h.parallel_for(ndr, [=](s::nd_item<3> it) {
      const int nco = it.get_group(0);
      const int gy  = it.get_group(1);
      const int gx  = it.get_group(2);
      const int ly  = it.get_local_id(1);
      const int lx  = it.get_local_id(2);

      const int n  = nco / (int)Co;
      const int co = nco % (int)Co;
      if (n >= (int)N || co >= (int)Co) return;

      const int oy0 = gy * TH;
      const int ox0 = gx * TW;
      const int oy  = oy0 + ly;
      const int ox  = ox0 + lx;

      if (oy >= (int)Ho || ox >= (int)Wo) return;

      const float* x_n = x + (int64_t)n * CiHW;
      float* y_plane = y + ((int64_t)n*Co + co) * HoWo;

      float acc = (bias ? bias[co] : 0.f);
      const int in_y = oy - pad;
      const int in_x = ox - pad;

      #pragma unroll
      for (int ci0 = 0; ci0 < (int)Ci; ci0 += ko_step) {
        const int ci_end = std::min((int)Ci, ci0 + ko_step);
        
        for (int ci = ci0; ci < ci_end; ++ci) {
          const float* wptr = w + ((int64_t)co*Ci + ci) * 9;
          const float* x_ci = x_n + (int64_t)ci * HW;
          
          #pragma unroll
          for (int ky=0; ky<3; ++ky) {
            const int iy = in_y + ky;
            if (iy < 0 || iy >= (int)H) continue;
            const float* row_ptr = x_ci + (int64_t)iy * W;
            const int ix = in_x;
            
            // Try vec4 first (most common case)
            if (use_v4 && ix >= 0 && (ix + 3) < (int)W) {
              const float* p = row_ptr + ix;
              if (aligned_vec4_ptr(p)) {
                const auto v4 = *reinterpret_cast<const s::vec<float,4>*>(p);
                acc += dot3_v4_row(v4, wptr + ky*3);
                continue;
              }
            }
            #pragma unroll
            for (int kx=0; kx<3; ++kx) {
              const int jx = ix + kx;
              if (jx < 0 || jx >= (int)W) continue;
              acc = fma(row_ptr[jx], wptr[ky*3 + kx], acc);
            }
          }
        }
      }

      if (acc < 0.f) acc = 0.f;
      y_plane[(int64_t)oy*Wo + ox] = acc;
    });
  });
}

static void launch_conv3x3_global_stride2(
    s::queue& q, bool use_v4, int ko_step,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int64_t N, int64_t Ci, int64_t H, int64_t W,
    int64_t Co, int64_t Ho, int64_t Wo,
    int pad, int TH, int TW)
{
  const int64_t HW = H * W;
  const int64_t CiHW = Ci * HW;
  const int64_t HoWo = Ho * Wo;
  const s::range<3> lrange(1, TH, TW);
  const s::range<3> grange(N*Co, (Ho+TH-1)/TH, (Wo+TW-1)/TW);
  const s::nd_range<3> ndr(grange * lrange, lrange);

  q.submit([&](s::handler& h) {
    h.parallel_for(ndr, [=](s::nd_item<3> it) {
      const int nco = it.get_group(0);
      const int gy  = it.get_group(1);
      const int gx  = it.get_group(2);
      const int ly  = it.get_local_id(1);
      const int lx  = it.get_local_id(2);

      const int n  = nco / (int)Co;
      const int co = nco % (int)Co;
      if (n >= (int)N || co >= (int)Co) return;

      const int oy0 = gy * TH;
      const int ox0 = gx * TW;
      const int oy  = oy0 + ly;
      const int ox  = ox0 + lx;

      if (oy >= (int)Ho || ox >= (int)Wo) return;

      const float* x_n = x + (int64_t)n * CiHW;
      float* y_plane = y + ((int64_t)n*Co + co) * HoWo;

      float acc = (bias ? bias[co] : 0.f);
      const int in_y = oy*2 - pad;
      const int in_x = ox*2 - pad;

      #pragma unroll
      for (int ci0 = 0; ci0 < (int)Ci; ci0 += ko_step) {
        const int ci_end = std::min((int)Ci, ci0 + ko_step);
        
        for (int ci = ci0; ci < ci_end; ++ci) {
          const float* wptr = w + ((int64_t)co*Ci + ci) * 9;
          const float* x_ci = x_n + (int64_t)ci * HW;
          
          #pragma unroll
          for (int ky=0; ky<3; ++ky) {
            const int iy = in_y + ky;
            if (iy < 0 || iy >= (int)H) continue;
            const float* row_ptr = x_ci + (int64_t)iy * W;
            const int ix = in_x;
            if (use_v4 && ix >= 0 && (ix + 3) < (int)W) {
              const float* p = row_ptr + ix;
              if (aligned_vec4_ptr(p)) {
                const auto v4 = *reinterpret_cast<const s::vec<float,4>*>(p);
                acc += dot3_v4_row(v4, wptr + ky*3);
                continue;
              }
            }
            #pragma unroll
            for (int kx=0; kx<3; ++kx) {
              const int jx = ix + kx;
              if (jx < 0 || jx >= (int)W) continue;
              acc = fma(row_ptr[jx], wptr[ky*3 + kx], acc);
            }
          }
        }
      }

      if (acc < 0.f) acc = 0.f;
      y_plane[(int64_t)oy*Wo + ox] = acc;
    });
  });
}

static void launch_conv3x3_global_stride1_nhwc(
    s::queue& q, int block_step,
    const float* __restrict__ x,   // [N,H,W,Ci] (channels_last)
    const float* __restrict__ w,   // [Co,3,3,Ci]
    const float* __restrict__ bias,
    float* __restrict__ y,         // NCHW output
    int64_t N, int64_t Ci_eff, int64_t H, int64_t W,
    int64_t Co, int64_t Ho, int64_t Wo,
    int pad, int TH, int TW,
    uint64_t* stats_ptr)
{
  const int64_t row_stride = W * Ci_eff;
  const int64_t plane_stride = H * row_stride;
  const int64_t HoWo = Ho * Wo;
  const int64_t w_stride_co = 9 * Ci_eff;
  const s::range<3> lrange(1, TH, TW);
  const s::range<3> grange(N*Co, (Ho+TH-1)/TH, (Wo+TW-1)/TW);
  const s::nd_range<3> ndr(grange * lrange, lrange);
  const int step = std::max(4, block_step);

  q.submit([&](s::handler& h) {
    h.parallel_for(ndr, [=](s::nd_item<3> it) {
      const int nco = it.get_group(0);
      const int gy  = it.get_group(1);
      const int gx  = it.get_group(2);
      const int ly  = it.get_local_id(1);
      const int lx  = it.get_local_id(2);

      const int n  = nco / (int)Co;
      const int co = nco % (int)Co;
      if (n >= (int)N || co >= (int)Co) return;

      const int oy0 = gy * TH;
      const int ox0 = gx * TW;
      const int oy  = oy0 + ly;
      const int ox  = ox0 + lx;
      if (oy >= (int)Ho || ox >= (int)Wo) return;

      const float* x_n = x + (int64_t)n * plane_stride;
      float* y_plane = y + ((int64_t)n*Co + co) * HoWo;
      const float* w_co = w + (int64_t)co * w_stride_co;
      const bool enable_stats = stats_ptr != nullptr;

      float acc = (bias ? bias[co] : 0.f);
      const int in_y = oy - pad;
      const int in_x = ox - pad;

      for (int ci0 = 0; ci0 < (int)Ci_eff; ci0 += step) {
        const int chunk = std::min(step, (int)Ci_eff - ci0);
        for (int ky=0; ky<3; ++ky) {
          const int iy = in_y + ky;
          if (iy < 0 || iy >= (int)H) continue;
          const float* row_ptr = x_n + (int64_t)iy * row_stride;
            const float* w_row = w_co + ky*3*Ci_eff;
          for (int kx=0; kx<3; ++kx) {
            const int ix = in_x + kx;
            if (ix < 0 || ix >= (int)W) continue;
            const float* pix = row_ptr + (int64_t)ix * Ci_eff + ci0;
            const float* w_pix = w_row + kx*Ci_eff + ci0;
            int lane = 0;
            
            // Process vec16 chunks
            for (; lane + 15 < chunk; lane += 16) {
              const float* px = pix + lane;
              const float* pw = w_pix + lane;
              
              if (aligned_vec16_ptr(px) && aligned_vec16_ptr(pw)) {
                if (enable_stats) { atomic64(stats_ptr[0]).fetch_add(1); }
                const auto vx = *reinterpret_cast<const s::vec<float,16>*>(px);
                const auto vw = *reinterpret_cast<const s::vec<float,16>*>(pw);
                acc += dot16(vx, vw);
              } else if (aligned_vec8_ptr(px) && aligned_vec8_ptr(pw)) {
                // Fallback to vec8 if vec16 not aligned
                const auto vx8a = *reinterpret_cast<const s::vec<float,8>*>(px);
                const auto vw8a = *reinterpret_cast<const s::vec<float,8>*>(pw);
                acc += dot8(vx8a, vw8a);
                const auto vx8b = *reinterpret_cast<const s::vec<float,8>*>(px + 8);
                const auto vw8b = *reinterpret_cast<const s::vec<float,8>*>(pw + 8);
                acc += dot8(vx8b, vw8b);
              } else {
                if (enable_stats) { atomic64(stats_ptr[1]).fetch_add(1); }
                #pragma unroll
                for (int u=0; u<16; ++u) {
                  acc = fma(px[u], pw[u], acc);
                }
              }
            }
            // Process vec8 chunks
            for (; lane + 7 < chunk; lane += 8) {
              const float* px = pix + lane;
              const float* pw = w_pix + lane;
              if (aligned_vec8_ptr(px) && aligned_vec8_ptr(pw)) {
                const auto vx = *reinterpret_cast<const s::vec<float,8>*>(px);
                const auto vw = *reinterpret_cast<const s::vec<float,8>*>(pw);
                acc += dot8(vx, vw);
              } else {
                #pragma unroll
                for (int u=0; u<8; ++u) {
                  acc = fma(px[u], pw[u], acc);
                }
              }
            }
            // Process vec4 chunks
            for (; lane + 3 < chunk; lane += 4) {
              acc = fma(pix[lane + 0], w_pix[lane + 0], acc);
              acc = fma(pix[lane + 1], w_pix[lane + 1], acc);
              acc = fma(pix[lane + 2], w_pix[lane + 2], acc);
              acc = fma(pix[lane + 3], w_pix[lane + 3], acc);
            }
            // Process remaining
            for (; lane < chunk; ++lane) {
              acc = fma(pix[lane], w_pix[lane], acc);
            }
          }
        }
      }

      if (acc < 0.f) acc = 0.f;
      y_plane[(int64_t)oy*Wo + ox] = acc;
    });
  });
}

static void launch_conv3x3_global_stride2_nhwc(
    s::queue& q, int block_step,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int64_t N, int64_t Ci_eff, int64_t H, int64_t W,
    int64_t Co, int64_t Ho, int64_t Wo,
    int pad, int TH, int TW,
    uint64_t* stats_ptr)
{
  const int64_t row_stride = W * Ci_eff;
  const int64_t plane_stride = H * row_stride;
  const int64_t HoWo = Ho * Wo;
  const int64_t w_stride_co = 9 * Ci_eff;
  const s::range<3> lrange(1, TH, TW);
  const s::range<3> grange(N*Co, (Ho+TH-1)/TH, (Wo+TW-1)/TW);
  const s::nd_range<3> ndr(grange * lrange, lrange);
  const int step = std::max(4, block_step);

  q.submit([&](s::handler& h) {
    h.parallel_for(ndr, [=](s::nd_item<3> it) {
      const int nco = it.get_group(0);
      const int gy  = it.get_group(1);
      const int gx  = it.get_group(2);
      const int ly  = it.get_local_id(1);
      const int lx  = it.get_local_id(2);

      const int n  = nco / (int)Co;
      const int co = nco % (int)Co;
      if (n >= (int)N || co >= (int)Co) return;

      const int oy0 = gy * TH;
      const int ox0 = gx * TW;
      const int oy  = oy0 + ly;
      const int ox  = ox0 + lx;
      if (oy >= (int)Ho || ox >= (int)Wo) return;

      const float* x_n = x + (int64_t)n * plane_stride;
      float* y_plane = y + ((int64_t)n*Co + co) * HoWo;
      const float* w_co = w + (int64_t)co * w_stride_co;
      const bool enable_stats = stats_ptr != nullptr;

      float acc = (bias ? bias[co] : 0.f);
      const int in_y = oy*2 - pad;
      const int in_x = ox*2 - pad;

      for (int ci0 = 0; ci0 < (int)Ci_eff; ci0 += step) {
        const int chunk = std::min(step, (int)Ci_eff - ci0);
        for (int ky=0; ky<3; ++ky) {
          const int iy = in_y + ky;
          if (iy < 0 || iy >= (int)H) continue;
          const float* row_ptr = x_n + (int64_t)iy * row_stride;
            const float* w_row = w_co + ky*3*Ci_eff;
          for (int kx=0; kx<3; ++kx) {
            const int ix = in_x + kx;
            if (ix < 0 || ix >= (int)W) continue;
            const float* pix = row_ptr + (int64_t)ix * Ci_eff + ci0;
            const float* w_pix = w_row + kx*Ci_eff + ci0;
            int lane = 0;
            
            // Process vec16 chunks
            for (; lane + 15 < chunk; lane += 16) {
              const float* px = pix + lane;
              const float* pw = w_pix + lane;
              
              if (aligned_vec16_ptr(px) && aligned_vec16_ptr(pw)) {
                if (enable_stats) { atomic64(stats_ptr[0]).fetch_add(1); }
                const auto vx = *reinterpret_cast<const s::vec<float,16>*>(px);
                const auto vw = *reinterpret_cast<const s::vec<float,16>*>(pw);
                acc += dot16(vx, vw);
              } else if (aligned_vec8_ptr(px) && aligned_vec8_ptr(pw)) {
                // Fallback to vec8 if vec16 not aligned
                const auto vx8a = *reinterpret_cast<const s::vec<float,8>*>(px);
                const auto vw8a = *reinterpret_cast<const s::vec<float,8>*>(pw);
                acc += dot8(vx8a, vw8a);
                const auto vx8b = *reinterpret_cast<const s::vec<float,8>*>(px + 8);
                const auto vw8b = *reinterpret_cast<const s::vec<float,8>*>(pw + 8);
                acc += dot8(vx8b, vw8b);
              } else {
                if (enable_stats) { atomic64(stats_ptr[1]).fetch_add(1); }
                #pragma unroll
                for (int u=0; u<16; ++u) {
                  acc = fma(px[u], pw[u], acc);
                }
              }
            }
            // Process vec8 chunks
            for (; lane + 7 < chunk; lane += 8) {
              const float* px = pix + lane;
              const float* pw = w_pix + lane;
              if (aligned_vec8_ptr(px) && aligned_vec8_ptr(pw)) {
                const auto vx = *reinterpret_cast<const s::vec<float,8>*>(px);
                const auto vw = *reinterpret_cast<const s::vec<float,8>*>(pw);
                acc += dot8(vx, vw);
              } else {
                #pragma unroll
                for (int u=0; u<8; ++u) {
                  acc = fma(px[u], pw[u], acc);
                }
              }
            }
            // Process vec4 chunks
            for (; lane + 3 < chunk; lane += 4) {
              acc = fma(pix[lane + 0], w_pix[lane + 0], acc);
              acc = fma(pix[lane + 1], w_pix[lane + 1], acc);
              acc = fma(pix[lane + 2], w_pix[lane + 2], acc);
              acc = fma(pix[lane + 3], w_pix[lane + 3], acc);
            }
            // Process remaining
            for (; lane < chunk; ++lane) {
              acc = fma(pix[lane], w_pix[lane], acc);
            }
          }
        }
      }

      if (acc < 0.f) acc = 0.f;
      y_plane[(int64_t)oy*Wo + ox] = acc;
    });
  });
}

// ============================================================================
// IM2ROW (row-pack, double-buffered across rows)
// ============================================================================
static void launch_conv3x3_im2row_stride1(
    s::queue& q, int ko_step,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int64_t N, int64_t Ci, int64_t H, int64_t W,
    int64_t Co, int64_t Ho, int64_t Wo,
    int pad, int TH, int TW)
{
  const s::range<3> lrange(1, TH, TW);
  const s::range<3> grange(N*Co, (Ho+TH-1)/TH, (Wo+TW-1)/TW);
  const s::nd_range<3> ndr(grange * lrange, lrange);

  q.submit([&](s::handler& h) {
    s::local_accessor<float, 1> rowbuf0((size_t)TH * (size_t)(TW+2) * (size_t)ko_step, h);
    s::local_accessor<float, 1> rowbuf1((size_t)TH * (size_t)(TW+2) * (size_t)ko_step, h);

    h.parallel_for(ndr, [=](s::nd_item<3> it) {
      const int nco = it.get_group(0);
      const int gy  = it.get_group(1);
      const int gx  = it.get_group(2);
      const int th  = it.get_local_id(1);
      const int tw  = it.get_local_id(2);

      const int n  = nco / (int)Co;
      const int co = nco % (int)Co;

      const int oy0 = gy * TH;
      const int ox0 = gx * TW;
      const int oy  = oy0 + th;
      const int ox  = ox0 + tw;

      if (oy >= (int)Ho || ox >= (int)Wo) return;

      float acc = (bias ? bias[co] : 0.f);
      const int in_y = oy - pad;
      const int in_x = ox - pad;

      for (int ci0 = 0; ci0 < (int)Ci; ci0 += ko_step) {
        const int ci_max = sycl::min(ko_step, (int)Ci - ci0);

        // NOTE: local_accessors appear const inside device lambda; take as const-ref.
        auto prefetch_row = [&](int ky, const s::local_accessor<float,1>& buf){
          const int iy = in_y + ky;
          const size_t base = (size_t)th * (size_t)(TW+2) * (size_t)ko_step;
          const int ix_l = in_x + 0;      // kx=0
          const int ix_c = in_x + 1;      // kx=1
          const int ix_r = in_x + 2;      // kx=2

          // center column for this thread (unique)
          for (int k=0; k<ci_max; ++k) {
            const int ci = ci0 + k;
            float vc = 0.f;
            if (iy>=0 && iy<(int)H && ix_c>=0 && ix_c<(int)W)
              vc = x[XIDX4(n,ci,iy,ix_c,Ci,H,W)];
            buf[ base + (size_t)(tw + 1)*(size_t)ko_step + (size_t)k ] = vc;
          }
          // left halo by tw==0
          if (tw == 0) {
            for (int k=0; k<ci_max; ++k) {
              const int ci = ci0 + k;
              float v0 = 0.f;
              if (iy>=0 && iy<(int)H && ix_l>=0 && ix_l<(int)W)
                v0 = x[XIDX4(n,ci,iy,ix_l,Ci,H,W)];
              buf[ base + (size_t)0*(size_t)ko_step + (size_t)k ] = v0;
            }
          }
          // right halo by tw==TW-1
          if (tw == (TW-1)) {
            for (int k=0; k<ci_max; ++k) {
              const int ci = ci0 + k;
              float v2 = 0.f;
              if (iy>=0 && iy<(int)H && ix_r>=0 && ix_r<(int)W)
                v2 = x[XIDX4(n,ci,iy,ix_r,Ci,H,W)];
              buf[ base + (size_t)(TW + 1)*(size_t)ko_step + (size_t)k ] = v2;
            }
          }
        };

        prefetch_row(0, rowbuf0);
        it.barrier(s::access::fence_space::local_space);

        for (int ky=0; ky<3; ++ky) {
          if (ky<2) prefetch_row(ky+1, (ky&1)? rowbuf0 : rowbuf1);
          it.barrier(s::access::fence_space::local_space);

          const size_t base = (size_t)th * (size_t)(TW+2) * (size_t)ko_step;
          const s::local_accessor<float,1>& cur = (ky&1)? rowbuf1 : rowbuf0;
          const size_t col_l = base + (size_t)tw * (size_t)ko_step;
          const size_t col_c = col_l + (size_t)ko_step;
          const size_t col_r = col_c + (size_t)ko_step;

          for (int k=0; k<ci_max; ++k) {
            const int ci = ci0 + k;
            const float v0 = cur[col_l + k];
            const float v1 = cur[col_c + k];
            const float v2 = cur[col_r + k];
            const float* w_ci = w + ((int64_t)co*Ci + ci) * 9 + ky*3;
            acc = fma(v0, w_ci[0], acc);
            acc = fma(v1, w_ci[1], acc);
            acc = fma(v2, w_ci[2], acc);
          }
        }
        it.barrier(s::access::fence_space::local_space);
      }

      if (acc < 0.f) acc = 0.f;
      const int64_t yoff = ((int64_t)n*Co + co)*Ho*Wo + (int64_t)oy*Wo + ox;
      y[yoff] = acc;
    });
  });
}

static void launch_conv3x3_im2row_stride2(
    s::queue& q, int ko_step,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int64_t N, int64_t Ci, int64_t H, int64_t W,
    int64_t Co, int64_t Ho, int64_t Wo,
    int pad, int TH, int TW)
{
  const int64_t HoWo = Ho * Wo;
  const int64_t CiTiles = (Ci + ko_step - 1) / ko_step;
  const int vec = 8;
  const int64_t tile_elems = (int64_t)(TH + 2) * (TW + 2) * ko_step;
  const s::range<3> lrange(1, TH, TW);
  const s::range<3> grange(N*Co, (Ho+TH-1)/TH, (Wo+TW-1)/TW);
  const s::nd_range<3> ndr(grange * lrange, lrange);

  q.submit([&](s::handler& h) {
    s::local_accessor<float, 1> tile((size_t)tile_elems, h);
    h.parallel_for(ndr, [=](s::nd_item<3> it) {
      const int nco = it.get_group(0);
      const int gy  = it.get_group(1);
      const int gx  = it.get_group(2);
      const int ly  = it.get_local_id(1);
      const int lx  = it.get_local_id(2);

      const int n  = nco / (int)Co;
      const int co = nco % (int)Co;

      const int oy0 = gy * TH;
      const int ox0 = gx * TW;
      const int oy  = oy0 + ly;
      const int ox  = ox0 + lx;
      if (oy >= (int)Ho || ox >= (int)Wo) return;

      float acc = (bias ? bias[co] : 0.f);
      const int in_y0 = oy0*2 - pad;
      const int in_x0 = ox0*2 - pad;

      for (int ci_blk = 0; ci_blk < (int)CiTiles; ++ci_blk) {
        const int ci0 = ci_blk * ko_step;
        const int ci_max = sycl::min(ko_step, (int)Ci - ci0);

        for (int ty = ly; ty < TH + 2; ty += TH) {
          const int iy = in_y0 + ty;
          for (int tx = lx; tx < TW + 2; tx += TW) {
            const int ix = in_x0 + tx;
            for (int k = 0; k < ci_max; k += vec) {
              const int chunk = sycl::min(vec, ci_max - k);
              float vec_in[vec] = {0.f};
              if (iy >= 0 && iy < (int)(H*2) && ix >= 0 && ix < (int)(W*2)) {
                const int src_y = clamp_idx(iy, 0, (int)H*2 - 1);
                const int src_x = clamp_idx(ix, 0, (int)W*2 - 1);
                const int py = src_y / 2;
                const int px = src_x / 2;
                const float* base = x + (((int64_t)n*Ci + (ci0 + k)) * H + py) * W + px;
                for (int u=0; u<chunk; ++u) vec_in[u] = base[u*H*W];
              }
              for (int u=0; u<chunk; ++u) {
                tile[((size_t)ty*(TW+2) + (size_t)tx)*(size_t)ko_step + (size_t)(k+u)] = vec_in[u];
              }
            }
          }
        }
        it.barrier(s::access::fence_space::local_space);

        for (int ky=0; ky<3; ++ky) {
          const int trow = ly + ky;
          for (int kx=0; kx<3; ++kx) {
            const int tcol = lx + kx;
            const size_t toff = ((size_t)trow*(TW+2) + (size_t)tcol)*(size_t)ko_step;
            const float* wptr = w + ((int64_t)co*Ci + ci0) * 9 + ky*3 + kx;
            for (int k=0; k<ci_max; ++k) {
              acc = fma(tile[toff + k], wptr[k*9], acc);
            }
          }
        }
        it.barrier(s::access::fence_space::local_space);
      }

      acc = sycl::max(acc, 0.f);
      y[(int64_t)n*Co*HoWo + (int64_t)co*HoWo + (int64_t)oy*Wo + ox] = acc;
    });
  });
}

// ============================================================================
// SLM tile kernel (direct 3x3), stride=1 only
// ============================================================================
static void launch_conv3x3_slm_stride1(
    s::queue& q, int ko_step,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int64_t N, int64_t Ci, int64_t H, int64_t W,
    int64_t Co, int64_t Ho, int64_t Wo,
    int pad, int TH, int TW)
{
  const s::range<3> lrange(1, TH, TW);
  const s::range<3> grange(N*Co, (Ho+TH-1)/TH, (Wo+TW-1)/TW);
  const s::nd_range<3> ndr(grange*lrange, lrange);

  q.submit([&](s::handler& h){
    s::local_accessor<float, 1> tile((size_t)(TH+2)*(size_t)(TW+2)*(size_t)ko_step, h);
    h.parallel_for(ndr, [=](s::nd_item<3> it){
      const int nco = it.get_group(0);
      const int gy  = it.get_group(1);
      const int gx  = it.get_group(2);
      const int ly  = it.get_local_id(1);
      const int lx  = it.get_local_id(2);

      const int n  = nco / (int)Co;
      const int co = nco % (int)Co;

      const int oy0 = gy*TH;
      const int ox0 = gx*TW;
      const int oy  = oy0 + ly;
      const int ox  = ox0 + lx;
      if (oy >= (int)Ho || ox >= (int)Wo) return;

      float acc = (bias ? bias[co] : 0.f);
      const int in_y0 = oy0 - pad;
      const int in_x0 = ox0 - pad;

      for (int ci0 = 0; ci0 < (int)Ci; ci0 += ko_step) {
        const int ci_max = sycl::min(ko_step, (int)Ci - ci0);

        // Load halo tile into SLM cooperatively
        for (int ty = ly; ty < TH + 2; ty += TH) {
          const int iy = in_y0 + ty;
          for (int tx = lx; tx < TW + 2; tx += TW) {
            const int ix = in_x0 + tx;
            for (int k = 0; k < ci_max; ++k) {
              const int ci = ci0 + k;
              float v = 0.f;
              if (iy >= 0 && iy < (int)H && ix >= 0 && ix < (int)W)
                v = x[XIDX4(n, ci, iy, ix, Ci, H, W)];
              tile[((size_t)ty*(TW+2) + (size_t)tx)*(size_t)ko_step + (size_t)k] = v;
            }
          }
        }
        it.barrier(s::access::fence_space::local_space);

        // Compute 3x3 using SLM halo
        const int64_t wbase_ci0 = ((int64_t)co*Ci + ci0) * 9;
        for (int ky = 0; ky < 3; ++ky) {
          const int trow = ly + ky;
          for (int kx = 0; kx < 3; ++kx) {
            const int tcol = lx + kx;
            const size_t toff = ((size_t)trow*(TW+2) + (size_t)tcol)*(size_t)ko_step;
            const int64_t woff = wbase_ci0 + ky*3 + kx;
            for (int k = 0; k < ci_max; ++k) {
              acc = fma(tile[toff + k], w[woff + (int64_t)k*9], acc);
            }
          }
        }
        it.barrier(s::access::fence_space::local_space);
      }

      if (acc < 0.f) acc = 0.f;
      const int64_t yoff = ((int64_t)n*Co + co)*Ho*Wo + (int64_t)oy*Wo + ox;
      y[yoff] = acc;
    });
  });
}

static void launch_conv3x3_slm_stride2(
    s::queue& q, int ko_step,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int64_t N, int64_t Ci, int64_t H, int64_t W,
    int64_t Co, int64_t Ho, int64_t Wo,
    int pad, int TH, int TW)
{
  const int stride = 2;
  const int haloH = TH * stride + 2;
  const int haloW = TW * stride + 2;
  const s::range<3> lrange(1, TH, TW);
  const s::range<3> grange(N*Co, (Ho+TH-1)/TH, (Wo+TW-1)/TW);
  const s::nd_range<3> ndr(grange*lrange, lrange);

  q.submit([&](s::handler& h){
    s::local_accessor<float, 1> tile((size_t)haloH * (size_t)haloW * (size_t)ko_step, h);
    h.parallel_for(ndr, [=](s::nd_item<3> it){
      const int nco = it.get_group(0);
      const int gy  = it.get_group(1);
      const int gx  = it.get_group(2);
      const int ly  = it.get_local_id(1);
      const int lx  = it.get_local_id(2);

      const int n  = nco / (int)Co;
      const int co = nco % (int)Co;

      const int oy0 = gy*TH;
      const int ox0 = gx*TW;
      const int oy  = oy0 + ly;
      const int ox  = ox0 + lx;
      if (oy >= (int)Ho || ox >= (int)Wo) return;

      float acc = (bias ? bias[co] : 0.f);
      const int in_y0 = oy0*stride - pad;
      const int in_x0 = ox0*stride - pad;

      for (int ci0 = 0; ci0 < (int)Ci; ci0 += ko_step) {
        const int ci_max = sycl::min(ko_step, (int)Ci - ci0);

        for (int ty = ly; ty < haloH; ty += TH) {
          const int iy = in_y0 + ty;
          for (int tx = lx; tx < haloW; tx += TW) {
            const int ix = in_x0 + tx;
            for (int k = 0; k < ci_max; ++k) {
              const int ci = ci0 + k;
              float v = 0.f;
              if (iy >= 0 && iy < (int)H && ix >= 0 && ix < (int)W)
                v = x[XIDX4(n, ci, iy, ix, Ci, H, W)];
              tile[((size_t)ty*haloW + (size_t)tx)*(size_t)ko_step + (size_t)k] = v;
            }
          }
        }
        it.barrier(s::access::fence_space::local_space);

        const int64_t wbase_ci0 = ((int64_t)co*Ci + ci0) * 9;
        for (int ky = 0; ky < 3; ++ky) {
          const int trow = ly*stride + ky;
          for (int kx = 0; kx < 3; ++kx) {
            const int tcol = lx*stride + kx;
            const size_t toff = ((size_t)trow*haloW + (size_t)tcol)*(size_t)ko_step;
            const int64_t woff = wbase_ci0 + ky*3 + kx;
            for (int k = 0; k < ci_max; ++k) {
              acc = fma(tile[toff + k], w[woff + (int64_t)k*9], acc);
            }
          }
        }
        it.barrier(s::access::fence_space::local_space);
      }

      if (acc < 0.f) acc = 0.f;
      const int64_t yoff = ((int64_t)n*Co + co)*Ho*Wo + (int64_t)oy*Wo + ox;
      y[yoff] = acc;
    });
  });
}

// ============================================================================
// Winograd F(2x2,3x3), stride=1, pad=1
// ============================================================================
static void launch_conv3x3_winograd_f23_stride1(
    s::queue& q,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int64_t N, int64_t Ci, int64_t H, int64_t W,
    int64_t Co, int64_t Ho, int64_t Wo)
{
  const int tilesY = (Ho + 1) / 2;
  const int tilesX = (Wo + 1) / 2;
  const s::range<3> grange(N*Co, tilesY, tilesX);
  const s::range<3> lrange(1,1,1);
  const s::nd_range<3> ndr(grange*lrange, lrange);

  q.submit([&](s::handler& h){
    h.parallel_for(ndr, [=](s::nd_item<3> it){
      const int nco = it.get_group(0);
      const int ty  = it.get_group(1);
      const int tx  = it.get_group(2);
      const int n   = nco / (int)Co;
      const int co  = nco % (int)Co;

      // Transform matrices
      const float Bt[4][4] = {
        { 1.f,  0.f, -1.f,  0.f},
        { 0.f,  1.f,  1.f,  0.f},
        { 0.f, -1.f,  1.f,  0.f},
        { 0.f,  1.f,  0.f, -1.f}
      };
      const float G[4][3] = {
        { 1.f,     0.f,     0.f},
        { 0.5f,   0.5f,   0.5f},
        { 0.5f,  -0.5f,   0.5f},
        { 0.f,     0.f,     1.f}
      };
      const float At[2][4] = {
        { 1.f,  1.f,  1.f,  0.f},
        { 0.f,  1.f, -1.f, -1.f}
      };

      const int oy0 = ty*2;
      const int ox0 = tx*2;
      if (oy0 >= (int)Ho || ox0 >= (int)Wo) return;

      float M[4][4];
      for (int i=0;i<4;++i) for (int j=0;j<4;++j) M[i][j]=0.f;

      for (int ci=0; ci<(int)Ci; ++ci) {
        // Input patch d (4x4) with implicit zero-pad
        float d[4][4];
        for (int i=0;i<4;++i){
          const int iy = oy0 + i - 1;
          for (int j=0;j<4;++j){
            const int ix = ox0 + j - 1;
            float v=0.f;
            if (iy>=0 && iy<(int)H && ix>=0 && ix<(int)W)
              v = x[XIDX4(n,ci,iy,ix,Ci,H,W)];
            d[i][j]=v;
          }
        }
        // V = Bt * d * B  (use Bt as B^T)
        float t1[4][4];
        for (int i=0;i<4;++i){
          for (int j=0;j<4;++j){
            t1[i][j] = Bt[i][0]*d[0][j] + Bt[i][1]*d[1][j] + Bt[i][2]*d[2][j] + Bt[i][3]*d[3][j];
          }
        }
        float V[4][4];
        for (int i=0;i<4;++i){
          for (int j=0;j<4;++j){
            V[i][j] = t1[i][0]*Bt[j][0] + t1[i][1]*Bt[j][1] + t1[i][2]*Bt[j][2] + t1[i][3]*Bt[j][3];
          }
        }
        // Filter transform U = G * g * G^T
        float g[3][3];
        const int64_t wbase = ((int64_t)co*Ci + ci)*9;
        g[0][0]=w[wbase+0]; g[0][1]=w[wbase+1]; g[0][2]=w[wbase+2];
        g[1][0]=w[wbase+3]; g[1][1]=w[wbase+4]; g[1][2]=w[wbase+5];
        g[2][0]=w[wbase+6]; g[2][1]=w[wbase+7]; g[2][2]=w[wbase+8];

        float t2[4][3];
        for (int i=0;i<4;++i){
          for (int j=0;j<3;++j){
            t2[i][j] = G[i][0]*g[0][j] + G[i][1]*g[1][j] + G[i][2]*g[2][j];
          }
        }
        float U[4][4];
        for (int i=0;i<4;++i){
          for (int j=0;j<4;++j){
            U[i][j] = t2[i][0]*G[j][0] + t2[i][1]*G[j][1] + t2[i][2]*G[j][2];
          }
        }
        // M += U ∘ V
        for (int i=0;i<4;++i){
          for (int j=0;j<4;++j){
            M[i][j] += U[i][j] * V[i][j];
          }
        }
      }

      // Y = At * M * A
      float t3[2][4];
      for (int i=0;i<2;++i){
        for (int j=0;j<4;++j){
          t3[i][j] = At[i][0]*M[0][j] + At[i][1]*M[1][j] + At[i][2]*M[2][j] + At[i][3]*M[3][j];
        }
      }
      float Yt[2][2];
      for (int i=0;i<2;++i){
        for (int j=0;j<2;++j){
          const float A0 = At[0][j];
          const float A1 = At[1][j];
          float v = t3[i][0]*A0 + t3[i][1]*A1 + t3[i][2]*(-A1) + t3[i][3]*(-A1);
          if (bias) v += bias[co];
          if (v < 0.f) v = 0.f; // ReLU
          Yt[i][j] = v;
        }
      }

      const int y0 = oy0, x0 = ox0;
      const int64_t ybase = ((int64_t)n*Co + co)*Ho*Wo;
      if (y0 < (int)Ho && x0 < (int)Wo)     y[ybase + (int64_t)y0*Wo + x0] = Yt[0][0];
      if (y0 < (int)Ho && x0+1 < (int)Wo)   y[ybase + (int64_t)y0*Wo + (x0+1)] = Yt[0][1];
      if (y0+1 < (int)Ho && x0 < (int)Wo)   y[ybase + (int64_t)(y0+1)*Wo + x0] = Yt[1][0];
      if (y0+1 < (int)Ho && x0+1 < (int)Wo) y[ybase + (int64_t)(y0+1)*Wo + (x0+1)] = Yt[1][1];
    });
  });
}

// ======================= Op (dispatch) =======================
static Tensor conv3x3_bn_relu_xpu(
    const Tensor& x, const Tensor& w,
    const c10::optional<Tensor>& bias_opt,
    const c10::optional<Tensor>& mean_opt,
    const c10::optional<Tensor>& var_opt,
    const c10::optional<Tensor>& gamma_opt,
    const c10::optional<Tensor>& beta_opt,
    int64_t stride, int64_t padding, double eps)
{
  (void)mean_opt; (void)var_opt; (void)gamma_opt; (void)beta_opt; (void)eps;
  check_input(x); check_input(w);
  TORCH_CHECK(w.size(2) == 3 && w.size(3) == 3, "Kernel must be 3x3.");
  TORCH_CHECK(stride == 1 || stride == 2, "Only stride 1 or 2 supported.");

  const auto N  = x.size(0);
  const auto Ci = x.size(1);
  const auto H  = x.size(2);
  const auto W  = x.size(3);
  const auto Co = w.size(0);

  const auto [Ho, Wo] = out_hw(H, W, 3, stride, padding);

  // Optional fast path: delegate to ATen/oneDNN (IPEX) when requested.
  const std::string backend = get_env_s("XPU_FUSED_BACKEND", "kernel"); // "kernel" or "aten"
  if (backend == "aten") {
    auto y_aten = at::conv2d(x, w, bias_opt.value_or(Tensor()), {stride, stride}, {padding, padding}, {1,1}, 1);
    return at::relu_(y_aten);
  }

  // (BN fold) prepare fused weights/bias if BN params are provided
  Tensor w_use = w;
  Tensor b_use = bias_opt.value_or(Tensor());
  const bool has_bn = mean_opt.has_value() && var_opt.has_value() && gamma_opt.has_value() && beta_opt.has_value();
  if (has_bn) {
    Tensor mean  = mean_opt.value();
    Tensor var   = var_opt.value();
    Tensor gamma = gamma_opt.value();
    Tensor beta  = beta_opt.value();
    TORCH_CHECK(mean.numel()==Co && var.numel()==Co && gamma.numel()==Co && beta.numel()==Co, "BN params must match out-channels");

    // Ensure on same device as weights
    if (mean.device()  != w.device())  mean  = mean.to(w.device());
    if (var.device()   != w.device())  var   = var.to(w.device());
    if (gamma.device() != w.device())  gamma = gamma.to(w.device());
    if (beta.device()  != w.device())  beta  = beta.to(w.device());

    // scale = gamma / sqrt(var + eps)
    auto scale = gamma * at::rsqrt(var + eps);
    // fold weights: w' = w * scale.view({Co,1,1,1})
    w_use = w * scale.view({Co,1,1,1});

    // base conv bias (0 if none)
    Tensor b0 = bias_opt.has_value() && bias_opt->defined() ? bias_opt.value() : at::zeros({Co}, w.options());
    if (b0.device() != w.device()) b0 = b0.to(w.device());

    // bias': beta + scale * (b0 - mean)
    b_use = beta + scale * (b0 - mean);
  }
  if (!w_use.is_contiguous()) {
    w_use = w_use.contiguous();
  }

  auto y = at::empty({N, Co, Ho, Wo}, x.options());
  const int Ci_pad16 = (Ci >= 16) ? (((int)Ci + 15) & ~15) : (int)Ci;
  const bool need_pad16 = (Ci_pad16 != (int)Ci);

  Tensor x_nchw_base = x.is_contiguous() ? x : x.contiguous();
  const float* x_ptr_nchw = x_nchw_base.data_ptr<float>();
  Tensor x_nhwc_buf;
  const float* x_ptr_nhwc = nullptr;
  auto get_x_nchw = [&]() -> const float* { return x_ptr_nchw; };
  auto get_x_nhwc = [&]() -> const float* {
    if (x_ptr_nhwc) return x_ptr_nhwc;
    x_nhwc_buf = x.contiguous(at::MemoryFormat::ChannelsLast);
    x_ptr_nhwc = x_nhwc_buf.data_ptr<float>();
    return x_ptr_nhwc;
  };
  Tensor x_pad_nchw_buf;
  Tensor x_pad_nhwc_buf;
  const float* x_ptr_nhwc_pad = nullptr;
  auto get_x_nhwc_vec = [&]() -> const float* {
    if (!need_pad16) return get_x_nhwc();
    if (x_ptr_nhwc_pad) return x_ptr_nhwc_pad;
    x_pad_nchw_buf = at::zeros({N, Ci_pad16, H, W}, x.options());
    x_pad_nchw_buf.narrow(1, 0, Ci).copy_(x_nchw_base);
    x_pad_nhwc_buf = x_pad_nchw_buf.contiguous(at::MemoryFormat::ChannelsLast);
    x_ptr_nhwc_pad = x_pad_nhwc_buf.data_ptr<float>();
    return x_ptr_nhwc_pad;
  };

  const float* w_ptr = w_use.data_ptr<float>();
  Tensor w_pad_buf;
  Tensor w_hwcn_buf;
  const float* w_ptr_hwcn = nullptr;
  auto get_w_hwcn16 = [&]() -> const float* {
    if (w_ptr_hwcn) return w_ptr_hwcn;
    Tensor source = w_use;
    if (need_pad16) {
      w_pad_buf = at::zeros({Co, Ci_pad16, 3, 3}, w.options());
      w_pad_buf.narrow(1, 0, Ci).copy_(w_use);
      source = w_pad_buf;
    }
    w_hwcn_buf = source.permute({0,2,3,1}).contiguous();
    w_ptr_hwcn = w_hwcn_buf.data_ptr<float>();
    return w_ptr_hwcn;
  };

  const float* b_ptr = (b_use.defined()) ? b_use.data_ptr<float>() : nullptr;
  float* y_ptr       = y.data_ptr<float>();

  auto q = getCurrentXPUStream().queue();
  const bool enable_vec16_stats = env_present("ENABLE_VEC16_STATS");
  struct Vec16StatsDeleter {
    s::queue q;
    void operator()(uint64_t* p) const { if (p) s::free(p, q); }
  };
  std::unique_ptr<uint64_t, Vec16StatsDeleter> vec16_stats;
  uint64_t* vec16_raw = nullptr;
  if (enable_vec16_stats) {
    uint64_t* buf = s::malloc_shared<uint64_t>(2, q);
    buf[0] = buf[1] = 0;
    vec16_stats = std::unique_ptr<uint64_t, Vec16StatsDeleter>(buf, Vec16StatsDeleter{q});
    vec16_raw = buf;
  }

  // Env controls
  const bool ko_env_override = env_present("XPU_FUSED_KO_STEP");
  int ko_step = ko_env_override ? std::max(1, get_env_i("XPU_FUSED_KO_STEP", 8))
                                : auto_ko_step(Ci);
  // Optimize KO_STEP for better vectorization and cache utilization
  if (!ko_env_override) {
    if (stride == 1 && Ci_pad16 >= 16) {
      ko_step = std::max(ko_step, 16);
    }
    // For large channels, prefer larger KO_STEP to improve cache line utilization
    if (Ci_pad16 >= 128) {
      ko_step = std::max(ko_step, 32);
    }
  }
  ko_step = std::max(1, std::min<int>(ko_step, Ci_pad16));

  const bool tile_s1_override = env_present("XPU_FUSED_TILE_S1");
  const bool tile_s2_override = env_present("XPU_FUSED_TILE_S2");
  const auto [TH_s1_env, TW_s1_env] = parse_tile_env("XPU_FUSED_TILE_S1", TILE_H_S1_DEF, TILE_W_S1_DEF);
  const auto [TH_s2_env, TW_s2_env] = parse_tile_env("XPU_FUSED_TILE_S2", TILE_H_S2_DEF, TILE_W_S2_DEF);
  const auto [TH_s1, TW_s1] = tile_s1_override ? std::make_pair(TH_s1_env, TW_s1_env)
                                               : auto_tile_hw(Ho, Wo, TH_s1_env, TW_s1_env);
  const auto [TH_s2, TW_s2] = tile_s2_override ? std::make_pair(TH_s2_env, TW_s2_env)
                                               : auto_tile_hw(Ho, Wo, TH_s2_env, TW_s2_env);
  const bool use_vec4_s1 = get_env_bool_01("XPU_FUSED_VEC4_S1", USE_VEC4_S1_DEF!=0);
  const bool use_vec4_s2 = get_env_bool_01("XPU_FUSED_VEC4_S2", USE_VEC4_S2_DEF!=0);
  const bool layout_override = env_present("XPU_FUSED_LAYOUT");
  std::string layout = get_env_s("XPU_FUSED_LAYOUT", "direct");   // direct|im2row|slm
  const std::string algo   = get_env_s("XPU_FUSED_ALGO",   "auto");     // auto|winograd|sgemm
  const bool force_global = get_env_bool_01("XPU_FUSED_FORCE_GLOBAL", false);
  const bool ko_s2_override = env_present("XPU_FUSED_KO_STEP_S2");
  int ko_step_s2 = ko_s2_override ? std::max(1, get_env_i("XPU_FUSED_KO_STEP_S2", 8))
                                  : auto_ko_step(Ci);
  // Optimize KO_STEP for stride=2
  if (!ko_s2_override) {
    if (Ci_pad16 >= 16) {
      ko_step_s2 = std::max(ko_step_s2, 8);
    }
    // For large channels with stride=2, prefer larger KO_STEP
    if (Ci_pad16 >= 128) {
      ko_step_s2 = std::max(ko_step_s2, 16);
    }
  }
  ko_step_s2 = std::max(1, ko_step_s2);
  const int ko_step_s2_im2row = std::min<int>(ko_step_s2, (int)Ci);
  const int ko_step_s2_direct = std::min<int>(ko_step_s2, Ci_pad16);

  if (stride == 1) {
    // Force global NCHW if XPU_FUSED_FORCE_GLOBAL is set (for probe/testing)
    if (force_global) {
      launch_conv3x3_global_stride1(q, use_vec4_s1, ko_step, get_x_nchw(), w_ptr, b_ptr, y_ptr,
                                    N, Ci, H, W, Co, Ho, Wo, (int)padding, TH_s1, TW_s1);
    }
    // Auto-select best kernel based on problem size
    // Allow auto-select even if layout is set to "direct" (user might want smart selection)
    else if (algo == "auto" && (!layout_override || layout == "direct")) {
      // Prefer Winograd for small channels and standard padding
      if ((int)padding == 1 && Ci <= 128 && Co <= 256 && Ho >= 8 && Wo >= 8) {
        launch_conv3x3_winograd_f23_stride1(q, get_x_nchw(), w_ptr, b_ptr, y_ptr,
                                            N, Ci, H, W, Co, Ho, Wo);
      }
      // Prefer NHWC vec16 for channels_last (much faster than NCHW)
      else if (x.is_contiguous(at::MemoryFormat::ChannelsLast) && Ci_pad16 >= 16) {
        const float* x_vec = get_x_nhwc_vec();
        const float* w_vec = get_w_hwcn16();
        launch_conv3x3_global_stride1_nhwc(q, ko_step, x_vec, w_vec, b_ptr, y_ptr,
                                           N, Ci_pad16, H, W, Co, Ho, Wo, (int)padding, TH_s1, TW_s1,
                                           vec16_raw);
      }
      // Prefer SLM for medium-sized problems with good tile fit (only if not channels_last)
      else if (!x.is_contiguous(at::MemoryFormat::ChannelsLast) && 
               TH_s1 * TW_s1 <= 256 && Ho >= TH_s1 && Wo >= TW_s1) {
        launch_conv3x3_slm_stride1(q, ko_step, get_x_nchw(), w_ptr, b_ptr, y_ptr,
                                   N, Ci, H, W, Co, Ho, Wo, (int)padding, TH_s1, TW_s1);
      }
      // Default to direct NCHW
      else {
        launch_conv3x3_global_stride1(q, use_vec4_s1, ko_step, get_x_nchw(), w_ptr, b_ptr, y_ptr,
                                      N, Ci, H, W, Co, Ho, Wo, (int)padding, TH_s1, TW_s1);
      }
    }
    else if (algo == "winograd" && (int)padding == 1) {
      launch_conv3x3_winograd_f23_stride1(q, get_x_nchw(), w_ptr, b_ptr, y_ptr,
                                          N, Ci, H, W, Co, Ho, Wo);
    }
    else if (layout == "slm") {
      launch_conv3x3_slm_stride1(q, ko_step, get_x_nchw(), w_ptr, b_ptr, y_ptr,
                                 N, Ci, H, W, Co, Ho, Wo, (int)padding, TH_s1, TW_s1);
    }
    else if (algo == "sgemm" || layout == "im2row") {
      launch_conv3x3_im2row_stride1(q, ko_step, get_x_nchw(), w_ptr, b_ptr, y_ptr,
                                    N, Ci, H, W, Co, Ho, Wo, (int)padding, TH_s1, TW_s1);
    }
    else {
      if (layout == "direct") {
        const float* x_vec = get_x_nhwc_vec();
        const float* w_vec = get_w_hwcn16();
        launch_conv3x3_global_stride1_nhwc(q, ko_step, x_vec, w_vec, b_ptr, y_ptr,
                                           N, Ci_pad16, H, W, Co, Ho, Wo, (int)padding, TH_s1, TW_s1,
                                           vec16_raw);
      } else {
        launch_conv3x3_global_stride1(q, use_vec4_s1, ko_step, get_x_nchw(), w_ptr, b_ptr, y_ptr,
                                      N, Ci, H, W, Co, Ho, Wo, (int)padding, TH_s1, TW_s1);
      }
    }
  } else { // stride == 2
    // Force global NCHW if XPU_FUSED_FORCE_GLOBAL is set (for probe/testing)
    if (force_global) {
      launch_conv3x3_global_stride2(q, use_vec4_s2, std::min<int>(ko_step_s2_direct, (int)Ci),
                                     get_x_nchw(), w_ptr, b_ptr, y_ptr,
                                     N, Ci, H, W, Co, Ho, Wo, (int)padding, TH_s2, TW_s2);
    }
    // Auto-select best kernel based on problem size
    // Allow auto-select even if layout is set to "direct"
    else if (algo == "auto" && (!layout_override || layout == "direct")) {
      // Prefer NHWC vec16 for channels_last (much faster)
      if (x.is_contiguous(at::MemoryFormat::ChannelsLast) && Ci_pad16 >= 16) {
        const float* x_vec = get_x_nhwc_vec();
        const float* w_vec = get_w_hwcn16();
        launch_conv3x3_global_stride2_nhwc(q, ko_step_s2_direct, x_vec, w_vec, b_ptr, y_ptr,
                                           N, Ci_pad16, H, W, Co, Ho, Wo, (int)padding, TH_s2, TW_s2,
                                           vec16_raw);
      }
      // Prefer im2row for medium/large spatial sizes (better for downsampling)
      else if (Ho >= 8 && Wo >= 8 && (Ho * Wo) >= 64) {
        launch_conv3x3_im2row_stride2(q, ko_step_s2_im2row, get_x_nchw(), w_ptr, b_ptr, y_ptr,
                                     N, Ci, H, W, Co, Ho, Wo, (int)padding, TH_s2, TW_s2);
      }
      // Prefer SLM for small spatial sizes with good tile fit
      else if (TH_s2 * TW_s2 <= 256 && Ho >= TH_s2 && Wo >= TW_s2) {
        launch_conv3x3_slm_stride2(q, ko_step_s2_im2row, get_x_nchw(), w_ptr, b_ptr, y_ptr,
                                   N, Ci, H, W, Co, Ho, Wo, (int)padding, TH_s2, TW_s2);
      }
      // Default to direct NCHW
      else {
        launch_conv3x3_global_stride2(q, use_vec4_s2, std::min<int>(ko_step_s2_direct, (int)Ci),
                                      get_x_nchw(), w_ptr, b_ptr, y_ptr,
                                      N, Ci, H, W, Co, Ho, Wo, (int)padding, TH_s2, TW_s2);
      }
    }
    else if (!layout_override && layout == "direct") {
      layout = "im2row";
    }
    if (algo == "sgemm" || layout == "im2row") {
      launch_conv3x3_im2row_stride2(q, ko_step_s2_im2row, get_x_nchw(), w_ptr, b_ptr, y_ptr,
                                    N, Ci, H, W, Co, Ho, Wo, (int)padding, TH_s2, TW_s2);
    } else {
      if (layout == "direct") {
        const float* x_vec = get_x_nhwc_vec();
        const float* w_vec = get_w_hwcn16();
        launch_conv3x3_global_stride2_nhwc(q, ko_step_s2_direct, x_vec, w_vec, b_ptr, y_ptr,
                                           N, Ci_pad16, H, W, Co, Ho, Wo, (int)padding, TH_s2, TW_s2,
                                           vec16_raw);
      } else if (layout == "slm") {
        launch_conv3x3_slm_stride2(q, ko_step_s2_im2row, get_x_nchw(), w_ptr, b_ptr, y_ptr,
                                   N, Ci, H, W, Co, Ho, Wo, (int)padding, TH_s2, TW_s2);
      } else {
        launch_conv3x3_global_stride2(q, use_vec4_s2, std::min<int>(ko_step_s2_direct, (int)Ci),
                                      get_x_nchw(), w_ptr, b_ptr, y_ptr,
                                      N, Ci, H, W, Co, Ho, Wo, (int)padding, TH_s2, TW_s2);
      }
    }
  }

  q.wait_and_throw();
  if (enable_vec16_stats && vec16_raw) {
    std::cout << "[vec16] aligned=" << vec16_raw[0] << " fallback=" << vec16_raw[1] << std::endl;
  }
  return y;
}

static Tensor conv3x3_bn_relu_xpu_dispatch(
    const Tensor& x, const Tensor& w,
    const c10::optional<Tensor>& bias_opt,
    const c10::optional<Tensor>& mean_opt,
    const c10::optional<Tensor>& var_opt,
    const c10::optional<Tensor>& gamma_opt,
    const c10::optional<Tensor>& beta_opt,
    int64_t stride, int64_t padding, double eps)
{
  return conv3x3_bn_relu_xpu(x, w, bias_opt, mean_opt, var_opt, gamma_opt, beta_opt,
                             stride, padding, eps);
}

// ======================= Registration =======================
TORCH_LIBRARY(custom_fused_ops, m) {
  m.def("conv3x3_bn_relu_xpu(Tensor x, Tensor w, Tensor? bias, Tensor? mean, Tensor? var, Tensor? gamma, Tensor? beta, int stride, int padding, float eps=1e-5) -> Tensor");
  m.impl("conv3x3_bn_relu_xpu", c10::DispatchKey::CPU, TORCH_FN(conv3x3_bn_relu_xpu));
}
TORCH_LIBRARY_IMPL(custom_fused_ops, XPU, m) {
  m.impl("conv3x3_bn_relu_xpu", conv3x3_bn_relu_xpu_dispatch);
}
TORCH_LIBRARY_IMPL(custom_fused_ops, PrivateUse1, m) {
  m.impl("conv3x3_bn_relu_xpu", conv3x3_bn_relu_xpu_dispatch);
}
TORCH_LIBRARY_IMPL(custom_fused_ops, Autograd, m) {
  m.impl("conv3x3_bn_relu_xpu", torch::CppFunction::makeFallthrough());
}
TORCH_LIBRARY_IMPL(custom_fused_ops, AutogradXPU, m) {
  m.impl("conv3x3_bn_relu_xpu", torch::CppFunction::makeFallthrough());
}
TORCH_LIBRARY_IMPL(custom_fused_ops, AutogradPrivateUse1, m) {
  m.impl("conv3x3_bn_relu_xpu", torch::CppFunction::makeFallthrough());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { /* empty */ }
