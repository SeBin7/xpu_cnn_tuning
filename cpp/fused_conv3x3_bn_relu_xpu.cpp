#include <ATen/ATen.h>
#include <torch/library.h>

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
} // namespace

at::Tensor conv3x3_bnfold_relu_usm_xpu(
  const at::Tensor& x,          // [N,C,H,W], float32
  const at::Tensor& w,          // [K,C,3,3], float32
  c10::optional<at::Tensor> b,  // [K], float32 (optional)
  const at::Tensor& mean,       // [K]
  const at::Tensor& var,        // [K]
  const at::Tensor& gamma,      // [K]
  const at::Tensor& beta,       // [K]
  double eps,
  c10::optional<at::Tensor> residual // [N,K,H,W], optional
)
{
  check_shapes(x,w,b,mean,var,gamma,beta);
  TORCH_CHECK(x.scalar_type() == at::kFloat && w.scalar_type() == at::kFloat,
              "x and w must be float32");
  TORCH_CHECK(mean.scalar_type() == at::kFloat && var.scalar_type() == at::kFloat &&
              gamma.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat,
              "BatchNorm tensors must be float32");
  TORCH_CHECK(!x.is_sparse() && !w.is_sparse(), "sparse tensors not supported");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous(),
              "Inputs must be contiguous NCHW for fast conv2d");
  auto K = (int)w.size(0);

  auto inv_std = (var + eps).rsqrt();
  auto scale = gamma * inv_std;                                // [K]
  at::Tensor Wf = w * scale.view({K, 1, 1, 1});                // [K,C,3,3]

  at::Tensor bias_folded = beta - mean * scale;                // [K]
  if (b.has_value()) {
    bias_folded = bias_folded + *b;
  }
  bias_folded = bias_folded.contiguous();

  at::Tensor y = at::conv2d(
      x,
      Wf,
      bias_folded,
      /*stride=*/{1, 1},
      /*padding=*/{1, 1},
      /*dilation=*/{1, 1},
      /*groups=*/1);

  if (residual.has_value() && residual->defined()) {
    TORCH_CHECK(residual->sizes() == y.sizes(), "residual shape mismatch");
    TORCH_CHECK(residual->device().is_xpu(), "residual must be on XPU");
    TORCH_CHECK(residual->dtype() == x.dtype(), "residual dtype mismatch");
    y.add_(*residual);
  }

  return at::relu_(y);
}

TORCH_LIBRARY(fused_ops, m) {
  m.def("conv3x3_bnfold_relu_usm_xpu(Tensor x, Tensor w, Tensor? b, Tensor mean, Tensor var, Tensor gamma, Tensor beta, float eps, Tensor? residual) -> Tensor");
}

TORCH_LIBRARY_IMPL(fused_ops, XPU, m) {
  m.impl("conv3x3_bnfold_relu_usm_xpu", conv3x3_bnfold_relu_usm_xpu);
}
