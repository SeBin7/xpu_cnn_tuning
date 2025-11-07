# Updated to keep channels_last end-to-end, avoid layout thrashing, and surface
# fused C++ kernel schedules via env. C++ kernel now supports automatic selection.
#
# Env knobs (examples):
#   ENABLE_XPU_FUSED=on|off|auto
#   XPU_FUSED_ALGO=auto|winograd|sgemm  (auto: smart selection, default)
#   XPU_FUSED_LAYOUT=direct|im2row|slm  (auto: let C++ choose if not set)
#   XPU_FUSED_KO_STEP=<int>            (channel block size, auto-tuned if not set)
#   XPU_FUSED_TILE_S1=<H>x<W>          (tile size for stride=1, e.g. "8x16")
#   XPU_FUSED_TILE_S2=<H>x<W>          (tile size for stride=2)
#   XPU_FUSED_MICRO_N=<int>            (micro-batch size)
#   XPU_FUSED_FORCE_GLOBAL=1           (debug: force global NCHW kernel)
#   XPU_FUSED_KERNEL=global|tiled      (legacy, use XPU_FUSED_LAYOUT instead)
#
# Build prerequisite: libcustom_conv3x3_bn_relu_xpu.so in ./build/

from __future__ import annotations
import os, time, pathlib
from contextlib import contextmanager
from typing import List
import torch, torch.nn as nn, torch.nn.functional as F

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SO_PATH = _ROOT / "build" / "libcustom_conv3x3_bn_relu_xpu.so"
_LIB_LOADED = False
_FUSED_OK = None


@contextmanager
def _temp_env(key: str, value: str | None):
    """Temporarily set/unset an env var for the scope of a 'with' block."""
    prev = os.environ.get(key, None)
    try:
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def _load_lib_once() -> bool:
    global _LIB_LOADED
    if _LIB_LOADED:
        return True
    try:
        if not _SO_PATH.exists():
            print(f"[fused] load failed: shared object not found at {_SO_PATH}")
            return False
        torch.ops.load_library(str(_SO_PATH))
        # schema sanity check
        try:
            overload = torch.ops.custom_fused_ops.conv3x3_bn_relu_xpu.default
            schema_obj = getattr(overload, "schema", None) or getattr(overload, "_schema", None)
            if schema_obj is None:
                raise AttributeError("schema attribute missing")
            schema = str(schema_obj)
        except AttributeError:
            print("[fused] load failed: conv3x3_bn_relu_xpu symbol not registered")
            return False
        if "int stride" not in schema or "int padding" not in schema:
            print(f"[fused] load failed: outdated operator schema -> {schema}")
            print("        Rebuild libcustom_conv3x3_bn_relu_xpu.so to pick up latest interface.")
            return False
        _LIB_LOADED = True
        return True
    except Exception as e:
        print(f"[fused] load failed: {e}")
        return False


def _cxx_fused(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor | None,
    mean: torch.Tensor,
    var: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    stride: int,
    padding: int,
    eps: float,
) -> torch.Tensor:
    """Call the fused C++ op. Keep channels_last end-to-end."""
    # Keep NHWC so the C++ kernel can choose blocked/vectorized paths.
    x_cl = x.contiguous(memory_format=torch.channels_last)
    w_contig = w.contiguous()  # (K,C,3,3)
    bias_contig = None if b is None else b.contiguous()
    mean_contig = None if mean is None else mean.contiguous()
    var_contig = None if var is None else var.contiguous()
    gamma_contig = None if gamma is None else gamma.contiguous()
    beta_contig = None if beta is None else beta.contiguous()

    y = torch.ops.custom_fused_ops.conv3x3_bn_relu_xpu(
        x_cl, w_contig, bias_contig, mean_contig, var_contig, gamma_contig, beta_contig,
        stride=int(stride), padding=int(padding), eps=float(eps)
    )
    # Return as channels_last to play nicely with the rest of the model
    return y.contiguous(memory_format=torch.channels_last)


def _probe_fused_once() -> bool:
    if not _load_lib_once():
        return False
    prev = os.environ.get("XPU_FUSED_FORCE_GLOBAL")
    try:
        os.environ["XPU_FUSED_FORCE_GLOBAL"] = "1"  # robust path for probing
        shapes = [(1,3,8,8,8,1,1), (1,8,8,8,16,1,1), (1,16,4,4,16,2,1)]
        for (N, Cin, H, W, Cout, s, p) in shapes:
            x = torch.randn(N, Cin, H, W, device="xpu", dtype=torch.float32)
            x = x.contiguous(memory_format=torch.channels_last)
            w = torch.randn(Cout, Cin, 3, 3, device="xpu", dtype=torch.float32)
            b = torch.zeros(Cout, device="xpu", dtype=torch.float32)
            m = torch.zeros(Cout, device="xpu", dtype=torch.float32)
            v = torch.ones (Cout, device="xpu", dtype=torch.float32)
            g = torch.ones (Cout, device="xpu", dtype=torch.float32)
            be= torch.zeros(Cout, device="xpu", dtype=torch.float32)

            y = _cxx_fused(x, w, b, m, v, g, be, s, p, 1e-5)

            invstd = torch.rsqrt(v + 1e-5)
            scale = g * invstd
            w_eff = w * scale.view(Cout, 1, 1, 1)
            b_eff = be - m * scale + b * scale
            ref = torch.relu(F.conv2d(x.contiguous(), w_eff, bias=b_eff, stride=s, padding=p))
            if not torch.allclose(y, ref, rtol=5e-3, atol=5e-3):
                diff = (y - ref).abs()
                print(f"[fused] probe failed: fused/reference mismatch "
                      f"(max_diff={diff.max().item():.4e}, mean_diff={diff.mean().item():.4e})")
                return False
        torch.xpu.synchronize()
        return True
    except Exception as e:
        print(f"[fused] probe failed → disabling fused. reason: {e}")
        return False
    finally:
        if prev is None: os.environ.pop("XPU_FUSED_FORCE_GLOBAL", None)
        else: os.environ["XPU_FUSED_FORCE_GLOBAL"] = prev


def probe_fused_fast() -> tuple[float, float]:
    if not torch.xpu.is_available() or not _load_lib_once():
        return float("inf"), float("inf")
    N, Cin, H, W, Cout = 32, 64, 32, 32, 64
    stride, padding, eps = 1, 1, 1e-5
    x = torch.randn(N, Cin, H, W, device="xpu").contiguous(memory_format=torch.channels_last)
    w = torch.randn(Cout, Cin, 3, 3, device="xpu")
    b = torch.zeros(Cout, device="xpu")
    mean = torch.zeros(Cout, device="xpu"); var = torch.ones(Cout, device="xpu")
    gamma= torch.ones (Cout, device="xpu"); beta= torch.zeros(Cout, device="xpu")
    invstd = torch.rsqrt(var + eps); scale = gamma * invstd
    w_eff = w * scale.view(Cout,1,1,1); bfold = beta - mean * scale + b * scale

    def _time(fn, iters=5):
        torch.xpu.synchronize(); t0=time.perf_counter()
        for _ in range(iters): fn()
        torch.xpu.synchronize(); return (time.perf_counter()-t0)/iters

    tf = _time(lambda: _cxx_fused(x,w,b,mean,var,gamma,beta,stride,padding,eps))
    tr = _time(lambda: F.relu(F.conv2d(x.contiguous(), w_eff, bias=bfold, stride=stride, padding=padding)))
    return tf, tr


def fused_available() -> bool:
    global _FUSED_OK
    if _FUSED_OK is not None:
        return _FUSED_OK
    enabled = os.environ.get("ENABLE_XPU_FUSED", "auto").lower()
    if enabled == "off":
        _FUSED_OK = False; print("[fused] decided: off (env)"); return _FUSED_OK
    if enabled == "on":
        _FUSED_OK = _probe_fused_once(); print(f"[fused] decided (env on): {'on' if _FUSED_OK else 'off'}"); return _FUSED_OK
    _FUSED_OK = _probe_fused_once(); print(f"[fused] decided: {'on' if _FUSED_OK else 'off'} (auto)"); return _FUSED_OK


def ensure_fused_decided(force: str = "auto") -> bool:
    val = (force or "auto").lower()
    if val == "off":
        os.environ["ENABLE_XPU_FUSED"] = "off"; print("[fused] decided: off (user)"); return False
    ok = _probe_fused_once(); os.environ["ENABLE_XPU_FUSED"] = "on" if ok else "off"
    print(
        f"[fused] decided (user on): {'on' if ok else 'off'}" if val == "on"
        else f"[fused] decided: {'on' if ok else 'off'} (auto)"
    )
    return ok


def _choose_schedule(stride: int, k_out: int) -> str:
    """
    Determine which kernel schedule will be used (for logging only).
    C++ kernel now handles automatic selection based on problem size.
    This function is kept for backward compatibility and logging.
    """
    # Check new env vars first
    algo = os.environ.get("XPU_FUSED_ALGO", "auto").lower()
    layout = os.environ.get("XPU_FUSED_LAYOUT", "").lower()
    
    if algo == "winograd":
        return "winograd"
    if algo == "sgemm" or layout == "im2row":
        return "im2row"
    if layout == "slm":
        return "slm"
    if layout == "direct":
        return "direct"
    
    # Legacy support
    pref = os.environ.get("XPU_FUSED_KERNEL", "").lower()
    if pref == "tiled":
        return "slm"
    if pref == "global":
        return "global"
    
    # Default: auto (C++ will choose)
    return "auto"


def _maybe_micro_batches(x: torch.Tensor, max_n: int) -> List[torch.Tensor]:
    if max_n <= 0 or x.shape[0] <= max_n:
        return [x]
    chunks: List[torch.Tensor] = []
    n = x.shape[0]
    for off in range(0, n, max_n):
        chunks.append(x[off: min(n, off+max_n)])
    return chunks


def _micro_batch_limit(stride: int) -> int:
    if stride == 2 and "XPU_FUSED_MICRO_N_S2" in os.environ:
        return int(os.environ.get("XPU_FUSED_MICRO_N_S2", "0"))
    return int(os.environ.get("XPU_FUSED_MICRO_N", "0"))


class _FusedConv3x3BNReLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, running_mean, running_var, gamma, beta, eps: float, stride: int, padding: int):
        # Keep channels_last for the op
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        if not w.is_contiguous():
            w = w.contiguous()
        ctx.has_bias = b is not None
        y = _cxx_fused(x, w, b, running_mean, running_var, gamma, beta, stride, padding, eps)
        y = y.contiguous(memory_format=torch.channels_last)
        # Save scale for backward (BN-folded conv weight)
        scale = gamma * torch.rsqrt(running_var + eps)
        ctx.save_for_backward(x, w, scale, y)
        ctx.stride = int(stride); ctx.padding = int(padding)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x_cl, w, scale, y = ctx.saved_tensors
        stride, padding = ctx.stride, ctx.padding
        # ReLU backward gate on pre-activation output
        grad_out_nchw = grad_out.contiguous()
        y_nchw = y.contiguous()
        zero = torch.zeros((), dtype=y_nchw.dtype, device=y_nchw.device)
        gate = torch.heaviside(y_nchw, zero)
        grad_pre_nchw = grad_out_nchw * gate.to(grad_out_nchw.dtype)

        # Grad w.r.t input/weight using ATen reference (with folded scale)
        K = w.shape[0]
        w_eff = w * scale.view(K,1,1,1)
        x_nchw = x_cl.contiguous()
        grad_x = torch.nn.grad.conv2d_input(x_nchw.shape, w_eff, grad_pre_nchw, stride=stride, padding=padding)
        grad_w_eff = torch.nn.grad.conv2d_weight(x_nchw, w_eff.shape, grad_pre_nchw, stride=stride, padding=padding)
        grad_w = grad_w_eff * scale.view(K,1,1,1)
        grad_b = grad_pre_nchw.sum(dim=(0,2,3)) * scale if ctx.has_bias else None

        # Return grad in channels_last to stay consistent
        grad_x = grad_x.contiguous(memory_format=torch.channels_last)
        return grad_x, grad_w, grad_b, None, None, None, None, None, None, None


class FusedConv3x3BNReLUTrainS(nn.Module):
    def __init__(self, C_in, C_out, eps=1e-5, bias=True, stride=1, padding=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(C_out, C_in, 3, 3) * 0.02)
        self.bias   = nn.Parameter(torch.zeros(C_out)) if bias else None
        self.register_buffer("running_mean", torch.zeros(C_out))
        self.register_buffer("running_var",  torch.ones(C_out))
        self.register_buffer("gamma",        torch.ones(C_out))
        self.register_buffer("beta",         torch.zeros(C_out))
        self.eps = float(eps); self.stride=int(stride); self.padding=int(padding)
        self._use_fused = None

    def _ensure_mode(self):
        if self._use_fused is None:
            self._use_fused = fused_available()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_mode()
        if self._use_fused and _load_lib_once():
            schedule = _choose_schedule(self.stride, self.weight.shape[0])
            micro_n = _micro_batch_limit(self.stride)
            if not getattr(self, "_once", False):
                algo = os.environ.get("XPU_FUSED_ALGO", "auto")
                layout = os.environ.get("XPU_FUSED_LAYOUT", "")
                layout_str = f", layout={layout}" if layout else ""
                print(
                    f"[fused] using C++ kernel (algo={algo}, schedule={schedule}{layout_str}, "
                    f"C_in={self.weight.shape[1]}, C_out={self.weight.shape[0]}, "
                    f"stride={self.stride}, pad={self.padding})"
                )
                self._once = True
            try:
                x_cl = x.contiguous(memory_format=torch.channels_last)
                # Don't force global if auto-selection is enabled (let C++ choose)
                # Only force if explicitly requested via XPU_FUSED_FORCE_GLOBAL
                force_global = os.environ.get("XPU_FUSED_FORCE_GLOBAL", "0") == "1"
                with _temp_env("XPU_FUSED_FORCE_GLOBAL", "1" if force_global else None):
                    # Micro batching works with global/direct/auto schedules
                    use_micro = micro_n > 0 and x_cl.shape[0] > micro_n
                    use_micro = use_micro and (schedule in ("global", "direct", "auto"))
                    if use_micro:
                        ys = []
                        for xc in _maybe_micro_batches(x_cl, micro_n):
                            ys.append(_FusedConv3x3BNReLUFn.apply(
                                xc, self.weight, self.bias,
                                self.running_mean, self.running_var,
                                self.gamma, self.beta,
                                self.eps, self.stride, self.padding
                            ))
                        y = torch.cat(ys, dim=0)
                    else:
                        y = _FusedConv3x3BNReLUFn.apply(
                            x_cl, self.weight, self.bias,
                            self.running_mean, self.running_var,
                            self.gamma, self.beta,
                            self.eps, self.stride, self.padding
                        )
                return y
            except Exception as e:
                print(f"[fused] runtime failure → fallback unfused. reason: {e}")
                self._use_fused = False

        # unfused path (reference)
        x_nchw = x.contiguous()  # NCHW ref math
        pre = F.conv2d(x_nchw, self.weight, bias=self.bias, stride=self.stride, padding=self.padding)
        invstd = 1.0 / torch.sqrt(self.running_var + self.eps)
        scale  = self.gamma * invstd
        y = (pre - self.running_mean.view(1,-1,1,1)) * scale.view(1,-1,1,1) + self.beta.view(1,-1,1,1)
        y = F.relu(y, inplace=False)
        return y.contiguous(memory_format=torch.channels_last)


def build_model(cfg):
    from .tinycifarnet import TinyCIFARNet
    m = cfg["model"]; t = cfg.get("train", {})
    return TinyCIFARNet(
        num_classes=int(m["num_classes"]),
        widths=tuple(m["widths"]),
        ffn_expand=int(m.get("ffn_expand", 4)),
        dropout=float(t.get("dropout", 0.1)),
        use_fused=True,
    )
