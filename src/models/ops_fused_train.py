import os
import time
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SO_PATH = _ROOT / "build" / "libcustom_conv3x3_bn_relu_xpu.so"
_LIB_LOADED = False
_FUSED_OK = None

def _load_lib_once():
    global _LIB_LOADED
    if _LIB_LOADED:
        return True
    try:
        if not _SO_PATH.exists():
            print(f"[fused] load failed: shared object not found at {_SO_PATH}")
            return False
        torch.ops.load_library(str(_SO_PATH))
        try:
            overload = torch.ops.custom_fused_ops.conv3x3_bn_relu_xpu.default
            schema_obj = getattr(overload, "schema", None)
            if schema_obj is None:
                schema_obj = getattr(overload, "_schema", None)
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

def _cxx_fused(x, w, b, mean, var, gamma, beta, stride: int, padding: int, eps: float):
    """Call C++ fused kernel (forward only)."""
    try:
        x_nchw = x.contiguous(memory_format=torch.contiguous_format)
        w_contig = w.contiguous(memory_format=torch.contiguous_format)
        bias_contig = None if b is None else b.contiguous()
        mean_contig = None if mean is None else mean.contiguous()
        var_contig = None if var is None else var.contiguous()
        gamma_contig = None if gamma is None else gamma.contiguous()
        beta_contig = None if beta is None else beta.contiguous()
        return torch.ops.custom_fused_ops.conv3x3_bn_relu_xpu(
            x_nchw, w_contig, bias_contig, mean_contig, var_contig, gamma_contig, beta_contig,
            stride=int(stride), padding=int(padding), eps=float(eps)
        )
    except RuntimeError as e:
        msg = str(e)
        if "No kernel named" in msg or "arguments for call are not valid" in msg:
            raise RuntimeError(
                "custom_fused_ops.conv3x3_bn_relu_xpu not registered with stride/padding signature. "
                "Rebuild libcustom_conv3x3_bn_relu_xpu.so (e.g. python fused_build_custom.py)."
            ) from e
        raise

def _probe_fused_once() -> bool:
    """
    Small correctness probe.
    Force GLOBAL path during probe to avoid subtle local-memory issues on some UMA configs.
    """
    if not _load_lib_once():
        return False
    prev_force = os.environ.get("XPU_FUSED_FORCE_GLOBAL")
    prev_kernel= os.environ.get("XPU_FUSED_KERNEL")
    try:
        os.environ["XPU_FUSED_FORCE_GLOBAL"] = "1"   # probe uses global kernel
        # (커널 선택 힌트) 사용자가 명시적으로 tiled를 고른 경우라도, 프로브는 global에서만 검사
        if prev_kernel is not None:
            os.environ["XPU_FUSED_KERNEL"] = "global"

        shapes = [
            (1, 3, 8,  8,  8, 1, 1),
            (1, 8, 8,  8, 16, 1, 1),
            (1,16, 4,  4, 16, 2, 1),  # stride-2 case
        ]
        for (N, Cin, H, W, Cout, s, p) in shapes:
            x = torch.randn(N, Cin, H, W, device="xpu", dtype=torch.float32, requires_grad=False)
            x = x.contiguous(memory_format=torch.channels_last)
            w = torch.randn(Cout, Cin, 3, 3, device="xpu", dtype=torch.float32, requires_grad=False)
            b = torch.zeros(Cout, device="xpu", dtype=torch.float32, requires_grad=False)
            m = torch.zeros(Cout, device="xpu", dtype=torch.float32)
            v = torch.ones (Cout, device="xpu", dtype=torch.float32)
            g = torch.ones (Cout, device="xpu", dtype=torch.float32)
            be= torch.zeros(Cout, device="xpu", dtype=torch.float32)

            y = _cxx_fused(x, w, b, m, v, g, be, s, p, 1e-5)

            # reference
            invstd = torch.rsqrt(v + 1e-5)
            scale = g * invstd
            w_eff = w * scale.view(Cout, 1, 1, 1)
            b_eff = be - m * scale + b * scale
            ref = torch.relu(torch.nn.functional.conv2d(x.contiguous(), w_eff, bias=b_eff, stride=s, padding=p))

            if not torch.allclose(y, ref, rtol=5e-3, atol=5e-3):
                diff = (y - ref).abs()
                print("[fused] probe failed: fused/reference mismatch "
                      f"(max_diff={diff.max().item():.4e}, mean_diff={diff.mean().item():.4e})")
                return False
        torch.xpu.synchronize()
        return True
    except Exception as e:
        print(f"[fused] probe failed → disabling fused. reason: {e}")
        return False
    finally:
        # restore envs
        if prev_force is None:
            os.environ.pop("XPU_FUSED_FORCE_GLOBAL", None)
        else:
            os.environ["XPU_FUSED_FORCE_GLOBAL"] = prev_force
        if prev_kernel is None:
            os.environ.pop("XPU_FUSED_KERNEL", None)
        else:
            os.environ["XPU_FUSED_KERNEL"] = prev_kernel

def probe_fused_fast():
    """
    Tiny benchmark helper used from train_xpu_overlap.py
    Returns (tf, tr) in seconds: fused forward vs reference forward.
    """
    if not torch.xpu.is_available():
        return float("inf"), float("inf")
    if not _load_lib_once():
        return float("inf"), float("inf")

    N, Cin, H, W, Cout = 32, 64, 32, 32, 64
    stride, padding = 1, 1
    eps = 1e-5

    x = torch.randn(N, Cin, H, W, device="xpu", dtype=torch.float32)
    x = x.contiguous(memory_format=torch.channels_last)
    w = torch.randn(Cout, Cin, 3, 3, device="xpu", dtype=torch.float32)
    b = torch.zeros(Cout, device="xpu", dtype=torch.float32)
    mean = torch.zeros(Cout, device="xpu", dtype=torch.float32)
    var  = torch.ones (Cout, device="xpu", dtype=torch.float32)
    gamma= torch.ones (Cout, device="xpu", dtype=torch.float32)
    beta = torch.zeros(Cout, device="xpu", dtype=torch.float32)

    invstd = torch.rsqrt(var + eps)
    scale = gamma * invstd
    w_eff = w * scale.view(Cout, 1, 1, 1)
    bfold = beta - mean * scale + b * scale

    def _time(fn, iters=5):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.xpu.synchronize()
        return (time.perf_counter() - t0) / iters

    x_nchw = x.contiguous()
    tf = _time(lambda: _cxx_fused(x, w, b, mean, var, gamma, beta, stride, padding, eps))
    tr = _time(lambda: F.relu(F.conv2d(x_nchw, w_eff, bias=bfold, stride=stride, padding=padding), inplace=False))
    return tf, tr

def fused_available() -> bool:
    """
    Decide once per process:
      - ENABLE_XPU_FUSED=off → force off
      - ENABLE_XPU_FUSED=on  → run probe (global) then enable if ok
      - ENABLE_XPU_FUSED=auto(default) → run probe and decide
    Note: kernel flavor (tiled/global) is selected in C++ via XPU_FUSED_KERNEL.
          현재 기본은 global 이며, SLM(tiled)을 시도하려면 명시적으로 환경변수를 설정해야 합니다.
    """
    global _FUSED_OK
    if (_FUSED_OK is not None):
        return _FUSED_OK
    enabled = os.environ.get("ENABLE_XPU_FUSED", "auto").lower()
    if enabled == "off":
        _FUSED_OK = False
        print("[fused] decided: off (env)")
        return _FUSED_OK
    if enabled == "on":
        _FUSED_OK = _probe_fused_once()
        print(f"[fused] decided (env on): {'on' if _FUSED_OK else 'off'}")
        return _FUSED_OK
    # auto
    _FUSED_OK = _probe_fused_once()
    print(f"[fused] decided: {'on' if _FUSED_OK else 'off'} (auto)")
    return _FUSED_OK

def ensure_fused_decided(force: str = "auto") -> bool:
    """
    Decide fused usage early.
    - "off": disable fused
    - "on" : run probe; keep on only if probe succeeds
    - "auto": probe and decide
    (Tip) To test tiled vs global at runtime:
        export XPU_FUSED_KERNEL=tiled|global
    """
    val = (force or "auto").lower()
    if val == "off":
        os.environ["ENABLE_XPU_FUSED"] = "off"
        print("[fused] decided: off (user)")
        return False
    if val == "on":
        ok = _probe_fused_once()
        os.environ["ENABLE_XPU_FUSED"] = "on" if ok else "off"
        print(f"[fused] decided (user on): {'on' if ok else 'off'}")
        return ok
    # 자동 모드
    ok = _probe_fused_once()
    os.environ["ENABLE_XPU_FUSED"] = "on" if ok else "off"
    print(f"[fused] decided: {'on' if ok else 'off'} (auto)")
    return ok

# -------------------- Autograd-safe fused layer --------------------

class _FusedConv3x3BNReLUFn(torch.autograd.Function):
    """
    Forward: custom C++ fused op (Conv3x3 + BN-fold(running stats) + ReLU)
    Backward: PyTorch reference grads.
    """

    @staticmethod
    def forward(ctx, x, w, b, running_mean, running_var, gamma, beta, eps: float, stride: int, padding: int):
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)

        ctx.has_bias = b is not None

        y = _cxx_fused(x, w, b, running_mean, running_var, gamma, beta, stride, padding, eps)
        y = y.contiguous(memory_format=torch.channels_last)

        # Save tensors for backward
        scale = gamma * torch.rsqrt(running_var + eps)
        ctx.save_for_backward(x, w, scale, y)
        ctx.stride = int(stride)
        ctx.padding = int(padding)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x_cl, w, scale, y = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding

        grad_out_nchw = grad_out.contiguous()
        y_nchw = y.contiguous()

        zero = torch.zeros((), dtype=y_nchw.dtype, device=y_nchw.device)
        gate = torch.heaviside(y_nchw, zero)
        grad_pre_nchw = grad_out_nchw * gate.to(grad_out_nchw.dtype)

        K = w.shape[0]
        w_eff = w * scale.view(K, 1, 1, 1)

        x_nchw = x_cl.contiguous()

        grad_x = torch.nn.grad.conv2d_input(
            x_nchw.shape, w_eff, grad_pre_nchw, stride=stride, padding=padding
        )
        grad_w_eff = torch.nn.grad.conv2d_weight(
            x_nchw, w_eff.shape, grad_pre_nchw, stride=stride, padding=padding
        )
        grad_w = grad_w_eff * scale.view(K,1,1,1)
        grad_b = grad_pre_nchw.sum(dim=(0,2,3)) if ctx.has_bias else None
        if grad_b is not None:
            grad_b = grad_b * scale

        grad_x = grad_x.contiguous(memory_format=torch.channels_last)

        return grad_x, grad_w, grad_b, None, None, None, None, None, None, None


class FusedConv3x3BNReLUTrainS(nn.Module):
    """Train-time fused block (forward fused, backward reference)."""
    def __init__(self, C_in, C_out, eps=1e-5, bias=True, stride=1, padding=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(C_out, C_in, 3, 3) * 0.02)
        self.bias   = nn.Parameter(torch.zeros(C_out)) if bias else None
        self.register_buffer("running_mean", torch.zeros(C_out))
        self.register_buffer("running_var",  torch.ones(C_out))
        self.register_buffer("gamma",        torch.ones(C_out))
        self.register_buffer("beta",         torch.zeros(C_out))
        self.eps = float(eps)
        self.stride  = int(stride)
        self.padding = int(padding)
        self._use_fused = None

    def _ensure_mode(self):
        if self._use_fused is not None:
            return
        self._use_fused = fused_available()

    def forward(self, x):
        self._ensure_mode()
        if self._use_fused and _load_lib_once():
            try:
                x_cl = x.contiguous(memory_format=torch.channels_last)
                return _FusedConv3x3BNReLUFn.apply(
                    x_cl, self.weight, self.bias,
                    self.running_mean, self.running_var,
                    self.gamma, self.beta,
                    self.eps, self.stride, self.padding
                )
            except Exception as e:
                print(f"[fused] runtime failure → fallback unfused. reason: {e}")
                self._use_fused = False

        # Unfused fallback (same stride/padding)
        x_nchw = x.contiguous()
        pre = F.conv2d(x_nchw, self.weight, bias=self.bias, stride=self.stride, padding=self.padding)
        invstd = 1.0 / torch.sqrt(self.running_var + self.eps)
        scale  = self.gamma * invstd
        y = (pre - self.running_mean.view(1,-1,1,1)) * scale.view(1,-1,1,1) + self.beta.view(1,-1,1,1)
        y = F.relu(y, inplace=False)
        return y.contiguous(memory_format=torch.channels_last)

def build_model(cfg):
    """
    models/__init__.py maps 'ops_tinycifarnet_sycl' to this factory.
    """
    from .tinycifarnet import TinyCIFARNet
    m = cfg["model"]
    t = cfg.get("train", {})
    num_classes = int(m["num_classes"])
    widths = tuple(m["widths"])
    ffn_expand = int(m.get("ffn_expand", 4))
    dropout = float(t.get("dropout", 0.1))
    return TinyCIFARNet(
        num_classes=num_classes,
        widths=widths,
        ffn_expand=ffn_expand,
        dropout=dropout,
        use_fused=True,
    )
