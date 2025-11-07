#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include <tuple>

#if __has_include(<ATen/xpu/XPUStream.h>)
  #include <ATen/xpu/XPUStream.h>
  using at::xpu::getCurrentXPUStream;
#elif __has_include(<c10/xpu/XPUStream.h>)
  #include <c10/xpu/XPUStream.h>
  using c10::xpu::getCurrentXPUStream;
#else
  #error "XPUStream header not found."
#endif

#include <sycl/sycl.hpp>

namespace {

static inline void check_shapes(
  const at::Tensor& x,
  const at::Tensor& w,
  const c10::optional<at::Tensor>& b,
  const at::Tensor& mean,
  const at::Tensor& var,
  const at::Tensor& gamma,
  const at::Tensor& beta)
{
  TORCH_CHECK(x.dim()==4, "x must be NCHW");
  TORCH_CHECK(w.dim()==4 && w.size(2)==3 && w.size(3)==3, "w must be [K,C,3,3]");
  TORCH_CHECK(mean.dim()==1 && var.dim()==1 && gamma.dim()==1 && beta.dim()==1, "bn params must be 1D");
  TORCH_CHECK(mean.size(0)==w.size(0) && var.size(0)==w.size(0) &&
              gamma.size(0)==w.size(0) && beta.size(0)==w.size(0), "bn param length must match out-channels");
  if (b.has_value()) TORCH_CHECK(b->dim()==1 && b->size(0)==w.size(0), "bias shape mismatch");
  TORCH_CHECK(x.device().is_xpu() && w.device().is_xpu() &&
              mean.device().is_xpu() && var.device().is_xpu() &&
              gamma.device().is_xpu() && beta.device().is_xpu(),
              "all tensors must be on XPU");
}

void launch_conv3x3_bnfold_relu_usm(
  sycl::queue& q,
  const float* X,   // [N,C,H,W]
  const float* Wf,  // [K,C,3,3]
  const float* Bf,  // [K]
  const float* Resid, // [N,K,H,W] or nullptr
  float* Y,         // [N,K,H,W]
  int N, int C, int H, int W, int K)
{
  size_t G = static_cast<size_t>(N)*K*H*W;
  q.parallel_for(sycl::range<1>(G), [=](sycl::id<1> tid) {
    size_t g = tid[0];
    int w = g % W;    g/=W;
    int h = g % H;    g/=H;
    int k = g % K;    g/=K;
    int n = static_cast<int>(g);

    float acc = Bf[k];

    // same padding=1, stride=1
    for (int c=0; c<C; ++c){
      for (int kh=0; kh<3; ++kh){
        int ih = h + kh - 1;
        if (ih<0 || ih>=H) continue;
        for (int kw=0; kw<3; ++kw){
          int iw = w + kw - 1;
          if (iw<0 || iw>=W) continue;
          size_t xoff = (((n*C + c)*H + ih)*W + iw);
          size_t woff = (((k*C + c)*3 + kh)*3 + kw);
          acc += X[xoff] * Wf[woff];
        }
      }
    }

    if (Resid){
      size_t roff = (((n*K + k)*H + h)*W + w);
      acc += Resid[roff];
    }
    if (acc < 0.0f) acc = 0.0f;

    size_t yoff = (((n*K + k)*H + h)*W + w);
    Y[yoff] = acc;
  }).wait();
}

void relu_backward_mask(
  sycl::queue& q,
  float* grad,
  const float* y,
  size_t total)
{
  q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> tid) {
    size_t idx = tid[0];
    grad[idx] = (y[idx] > 0.0f) ? grad[idx] : 0.0f;
  }).wait();
}

void launch_conv3x3_backward_input(
  sycl::queue& q,
  const float* grad,
  const float* Wf,
  float* grad_x,
  int N, int C, int H, int W, int K)
{
  size_t total = static_cast<size_t>(N) * C * H * W;
  q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> tid) {
    size_t g = tid[0];
    int w = g % W; g /= W;
    int h = g % H; g /= H;
    int c = g % C; g /= C;
    int n = static_cast<int>(g);

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      for (int kh = 0; kh < 3; ++kh) {
        int ho = h - kh + 1;
        if (ho < 0 || ho >= H) continue;
        for (int kw = 0; kw < 3; ++kw) {
          int wo = w - kw + 1;
          if (wo < 0 || wo >= W) continue;
          size_t goff = (((n*K + k)*H + ho)*W + wo);
          size_t woff = (((k*C + c)*3 + kh)*3 + kw);
          acc += grad[goff] * Wf[woff];
        }
      }
    }
    grad_x[tid[0]] = acc;
  }).wait();
}

void launch_conv3x3_backward_weight(
  sycl::queue& q,
  const float* x,
  const float* grad,
  float* grad_wf,
  int N, int C, int H, int W, int K)
{
  size_t total = static_cast<size_t>(K) * C * 3 * 3;
  q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> tid) {
    size_t g = tid[0];
    int kw = g % 3; g /= 3;
    int kh = g % 3; g /= 3;
    int c = g % C; g /= C;
    int k = static_cast<int>(g);

    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
      for (int ho = 0; ho < H; ++ho) {
        int h = ho + kh - 1;
        if (h < 0 || h >= H) continue;
        for (int wo = 0; wo < W; ++wo) {
          int w = wo + kw - 1;
          if (w < 0 || w >= W) continue;
          size_t goff = (((n*K + k)*H + ho)*W + wo);
          size_t xoff = (((n*C + c)*H + h)*W + w);
          acc += grad[goff] * x[xoff];
        }
      }
    }
    grad_wf[tid[0]] = acc;
  }).wait();
}

void launch_conv3x3_backward_bias(
  sycl::queue& q,
  const float* grad,
  float* grad_bf,
  int N, int K, int H, int W)
{
  q.parallel_for(sycl::range<1>(static_cast<size_t>(K)), [=](sycl::id<1> tid) {
    int k = static_cast<int>(tid[0]);
    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          size_t off = (((n*K + k)*H + h)*W + w);
          acc += grad[off];
        }
      }
    }
    grad_bf[k] = acc;
  }).wait();
}

} // namespace

at::Tensor conv3x3_bnfold_relu_usm_xpu(
  const at::Tensor& x,          // [N,C,H,W] float32
  const at::Tensor& w,          // [K,C,3,3] float32
  c10::optional<at::Tensor> b,  // [K] optional
  const at::Tensor& mean,       // [K]
  const at::Tensor& var,        // [K]
  const at::Tensor& gamma,      // [K]
  const at::Tensor& beta,       // [K]
  double eps,
  c10::optional<at::Tensor> residual // [N,K,H,W] optional
)
{
  check_shapes(x,w,b,mean,var,gamma,beta);
  TORCH_CHECK(x.scalar_type()==at::kFloat && w.scalar_type()==at::kFloat, "float32 only");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "contiguous NCHW");

  auto N = (int)x.size(0), C=(int)x.size(1), H=(int)x.size(2), W=(int)x.size(3), K=(int)w.size(0);
  auto opts = x.options();
  auto dev  = x.device();

  auto scale = (gamma / (var + eps).sqrt());             // [K]
  at::Tensor Wf = w * scale.view({K,1,1,1});                            // [K,C,3,3]
  at::Tensor b_post = b.has_value()? *b : at::zeros({K}, opts);
  at::Tensor Bf = beta - mean * scale + b_post;   // [K]

  at::Tensor y = at::empty({N,K,H,W}, opts);

  const float* Xptr = x.data_ptr<float>();
  const float* Wptr = Wf.data_ptr<float>();
  const float* Bptr = Bf.data_ptr<float>();
  float* Yptr       = y.data_ptr<float>();
  const float* Rptr = nullptr;

  if (residual.has_value() && residual->defined()){
    TORCH_CHECK(residual->sizes() == at::IntArrayRef({N,K,H,W}), "residual shape mismatch");
    TORCH_CHECK(residual->device().is_xpu() && residual->dtype()==x.dtype() && residual->is_contiguous(),
                "residual must be XPU float contiguous");
    Rptr = residual->data_ptr<float>();
  }

  auto q = getCurrentXPUStream().queue();
  launch_conv3x3_bnfold_relu_usm(q, Xptr, Wptr, Bptr, Rptr, Yptr, N,C,H,W,K);
  return y;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
conv3x3_bnfold_relu_backward_usm_xpu(
  const at::Tensor& grad_out,
  const at::Tensor& y,
  const at::Tensor& x,
  const at::Tensor& w,
  c10::optional<at::Tensor> b,
  const at::Tensor& mean,
  const at::Tensor& var,
  const at::Tensor& gamma,
  const at::Tensor& beta,
  double eps)
{
  check_shapes(x, w, b, mean, var, gamma, beta);

  auto grad = grad_out.contiguous();
  auto y_c = y.contiguous();
  auto x_c = x.contiguous();
  auto w_c = w.contiguous();

  TORCH_CHECK(grad.scalar_type()==at::kFloat && y_c.scalar_type()==at::kFloat, "float32 only");

  const int64_t N = x_c.size(0);
  const int64_t C = x_c.size(1);
  const int64_t H = x_c.size(2);
  const int64_t W = x_c.size(3);
  const int64_t K = w_c.size(0);

  auto opts = x_c.options();

  auto inv_std = (var + eps).rsqrt();
  auto scale = gamma * inv_std;
  at::Tensor Wf = (w_c * scale.view({K,1,1,1})).contiguous();

  at::Tensor grad_x = at::empty_like(x_c);
  at::Tensor grad_wf = at::empty_like(w_c);
  at::Tensor grad_bf = at::empty({K}, opts);

  auto q = getCurrentXPUStream().queue();

  relu_backward_mask(q, grad.data_ptr<float>(), y_c.data_ptr<float>(),
                     static_cast<size_t>(grad.numel()));

  launch_conv3x3_backward_input(
    q,
    grad.data_ptr<float>(),
    Wf.data_ptr<float>(),
    grad_x.data_ptr<float>(),
    static_cast<int>(N),
    static_cast<int>(C),
    static_cast<int>(H),
    static_cast<int>(W),
    static_cast<int>(K));

  launch_conv3x3_backward_weight(
    q,
    x_c.data_ptr<float>(),
    grad.data_ptr<float>(),
    grad_wf.data_ptr<float>(),
    static_cast<int>(N),
    static_cast<int>(C),
    static_cast<int>(H),
    static_cast<int>(W),
    static_cast<int>(K));

  launch_conv3x3_backward_bias(
    q,
    grad.data_ptr<float>(),
    grad_bf.data_ptr<float>(),
    static_cast<int>(N),
    static_cast<int>(K),
    static_cast<int>(H),
    static_cast<int>(W));

  at::Tensor grad_w = grad_wf * scale.view({K,1,1,1});
  at::Tensor grad_beta = grad_bf;
  at::Tensor grad_gamma =
      (grad_wf * w_c).sum({1,2,3}) * inv_std + grad_bf * (-mean * inv_std);

  at::Tensor grad_bias = b.has_value() ? grad_bf.clone() : at::Tensor();

  return {grad_x, grad_w, grad_bias, grad_gamma, grad_beta};
}

TORCH_LIBRARY(fused_ops, m) {
  m.def("conv3x3_bnfold_relu_usm_xpu(Tensor x, Tensor w, Tensor? b, Tensor mean, Tensor var, Tensor gamma, Tensor beta, float eps, Tensor? residual) -> Tensor");
  m.def("conv3x3_bnfold_relu_backward_usm_xpu(Tensor grad_out, Tensor y, Tensor x, Tensor w, Tensor? b, Tensor mean, Tensor var, Tensor gamma, Tensor beta, float eps) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(fused_ops, XPU, m) {
  m.impl("conv3x3_bnfold_relu_usm_xpu", conv3x3_bnfold_relu_usm_xpu);
  m.impl("conv3x3_bnfold_relu_backward_usm_xpu", conv3x3_bnfold_relu_backward_usm_xpu);
}
