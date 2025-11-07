from __future__ import annotations
import os, time, json, argparse, yaml, math
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# NOTE: we import models/ops *after* parsing CLI & wiring env so that
# any env-driven fused/XPU settings are visible to the extension loader.

# ------------------------
# Helpers
# ------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes", "true", "on", "1"):  return True
    if v in ("no",  "false", "off", "0"): return False
    if v == "auto":                          return "auto"
    raise argparse.ArgumentTypeError("expected on/off/true/false/auto")


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_cfg_defaults(cfg: dict) -> dict:
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    data = cfg.setdefault("data", {})
    dataset = cfg.setdefault("dataset", {})
    dataloader = cfg.setdefault("dataloader", {})
    misc = cfg.setdefault("misc", {})

    dataset.setdefault("name", data.get("name", "cifar10"))
    dataset.setdefault("root", data.get("root", "./data"))
    dataset.setdefault("img_size", data.get("img_size", 32))
    dataset.setdefault("download", data.get("download", True))
    dataset.setdefault("augment", {})
    if "mean" in data: dataset.setdefault("mean", tuple(data["mean"]))
    if "std"  in data: dataset.setdefault("std",  tuple(data["std"]))

    dataloader.setdefault("batch_size",           data.get("batch_size", 128))
    dataloader.setdefault("eval_batch_size",      data.get("eval_batch_size", dataloader["batch_size"]))
    dataloader.setdefault("num_workers",          data.get("num_workers", 4))
    dataloader.setdefault("pin_memory",           data.get("pin_memory", True))
    dataloader.setdefault("prefetch_factor",      data.get("prefetch_factor", 4))
    dataloader.setdefault("eval_prefetch_factor", dataloader["prefetch_factor"])
    dataloader.setdefault("persistent_workers",   data.get("persistent_workers", True))
    dataloader.setdefault("drop_last", False)
    dataloader.setdefault("shuffle", True)
    dataloader.setdefault("eval_shuffle", False)

    tr = cfg["train"]
    tr.setdefault("optimizer", "sgd")
    tr.setdefault("lr", 0.1)
    tr.setdefault("momentum", 0.9)
    tr.setdefault("weight_decay", 5e-4)
    tr.setdefault("epochs", 1)
    tr.setdefault("grad_accum", 1)
    tr.setdefault("label_smoothing", 0.0)

    misc.setdefault("channels_last", True)
    misc.setdefault("log_interval", 30)
    misc.setdefault("ckpt_dir", "./outputs/ckpts")
    misc.setdefault("ckpt_name", "tinycifarnet_xpu.pt")
    misc.setdefault("use_prefetcher", True)
    return cfg


def set_channels_last(model: nn.Module) -> nn.Module:
    return model.to(memory_format=torch.channels_last)


def save_checkpoint(path: str, model: nn.Module, epoch: int, best: float, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "best": float(best),
        "model": model.state_dict(),
        "extra": extra or {}
    }, path)


@torch.no_grad()
def top1_cpu(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.detach().argmax(1).to("cpu", non_blocking=False)
    t = targets.detach().to("cpu", non_blocking=False)
    return (preds == t).float().mean().item()


def to_host_scalar(t: torch.Tensor) -> float:
    x = t.detach()
    if x.device.type != "cpu":
        x = x.to("cpu", non_blocking=False)
    return float(x.item())


def cross_entropy_xpu_safe(logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    """UR 에러가 날 때 대체 가능한 CE. fp32로 강제하고 log_softmax 기반으로 계산.
    This mirrors PyTorch's CE while avoiding backend-specific fused paths.
    """
    logits = logits.float().contiguous()
    targets = targets.long().contiguous()
    if label_smoothing and label_smoothing > 0.0:
        n = logits.size(1)
        with torch.no_grad():
            true = torch.full_like(logits, label_smoothing / (n - 1))
            true.scatter(1, targets.unsqueeze(1), 1.0 - label_smoothing)
        logp = F.log_softmax(logits, dim=1)
        return (-(true * logp).sum(dim=1)).mean()
    return F.nll_loss(F.log_softmax(logits, dim=1), targets)


def ce_with_ur_fallback(ce: nn.Module, logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float) -> torch.Tensor:
    try:
        return ce(logits, targets)
    except RuntimeError as e:
        msg = str(e).lower()
        if "ur_result" in msg or "unified runtime" in msg or "native api failed" in msg:
            # Known Level-Zero/UR path → safe fallback
            return cross_entropy_xpu_safe(logits, targets, label_smoothing)
        raise


class LocalPrefetcher:
    """Simple pinned/USM-preferred prefetcher that overlaps H2D with compute."""
    def __init__(self, loader, device, channels_last: bool):
        self.loader = iter(loader)
        self.device = device
        self.channels_last = channels_last
        if device.type == "xpu":
            self.copy_stream = torch.xpu.Stream()
            self._stream_ctx = torch.xpu.stream
        elif device.type == "cuda":
            self.copy_stream = torch.cuda.Stream()
            self._stream_ctx = torch.cuda.stream
        else:
            self.copy_stream = None
            self._stream_ctx = None
        self._next = None
        self._preload()

    def _to_device(self, x, y):
        memfmt = torch.channels_last if self.channels_last else torch.contiguous_format
        return (
            x.to(self.device, memory_format=memfmt, non_blocking=True),
            y.to(self.device, non_blocking=True),
        )

    def _preload(self):
        try:
            x, y = next(self.loader)
        except StopIteration:
            self._next = None
            return
        if self.copy_stream is None:
            self._next = self._to_device(x, y)
            return
        with self._stream_ctx(self.copy_stream):
            self._next = self._to_device(x, y)

    def next(self):
        if self._next is None:
            return None, None
        if self.copy_stream is not None:
            if self.device.type == "xpu":
                torch.xpu.current_stream().wait_stream(self.copy_stream)
            else:
                torch.cuda.current_stream().wait_stream(self.copy_stream)
        batch = self._next
        self._preload()
        return batch


def amp_ctx(device_type: str, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device_type)


def profiler_context():
    out_dir = os.getenv("TORCH_PROFILER", "").strip()
    if not out_dir:
        return nullcontext()
    try:
        from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
    except Exception as exc:
        print(f"[profiler] disabled: {exc}")
        return nullcontext()
    activities = [ProfilerActivity.CPU]
    try:
        if torch.xpu.is_available():
            activities.append(ProfilerActivity.XPU)
    except AttributeError:
        pass
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    print(f"[profiler] writing trace to {path}")
    return profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=tensorboard_trace_handler(str(path)),
    )


def run_fused_warmup(steps: int):
    if steps <= 0:
        return
    try:
        from models import ops_fused_sycl_train as fused_mod
    except Exception as exc:
        print(f"[fused-warmup] skip: {exc}")
        return
    print(f"[fused-warmup] running {steps} probe iteration(s) to amortize first-step JIT/allocs.")
    for idx in range(steps):
        tf, tr = fused_mod.probe_fused_fast()
        if not (math.isfinite(tf) and math.isfinite(tr)):
            print("[fused-warmup] probe returned non-finite timings — aborting warmup.")
            break
        print(f"[fused-warmup] iter={idx+1}/{steps} fused={tf*1e3:.2f}ms ref={tr*1e3:.2f}ms")


def emit_fused_diagnosis(channels_last_on: bool, warmup_steps: int):
    layout = os.getenv("XPU_FUSED_LAYOUT", "direct")
    backend = os.getenv("XPU_FUSED_BACKEND", "kernel")
    kernel = os.getenv("XPU_FUSED_KERNEL", "auto")
    ko_step = os.getenv("XPU_FUSED_KO_STEP", "auto")
    tile_s1 = os.getenv("XPU_FUSED_TILE_S1", "auto")
    tile_s2 = os.getenv("XPU_FUSED_TILE_S2", "auto")
    fold = os.getenv("XPU_FUSED_FOLD_ON_CPU", "auto")
    print("[diag] ===== \"튜닝했는데 왜 더 느려졌지?\" 체크 포인트 =====")
    fmt_status = "OK" if channels_last_on else "경고: channels_last 비활성 → NHWC/NCHW 전환 비용 발생"
    print(f"[diag] (1) 메모리 포맷 : channels_last={channels_last_on} → {fmt_status}")
    print(f"[diag] (2) 글로벌 경로 : layout={layout}, backend={backend}, kernel={kernel}, KO_STEP={ko_step}")
    print(f"                ↳ direct/global일 때 vec4 접근이 정렬되어야 DRAM gather가 줄어듭니다.")
    print(f"[diag] (3) 타일/SLM    : TILE_S1={tile_s1}, TILE_S2={tile_s2}, BN_fold={fold}")
    print(f"                ↳ 타일을 너무 크게 잡으면 halo SLM 채움/배리어 비용이 증가합니다.")
    wmsg = "워밍업 비활성 (stream sync 비용이 첫 step에 몰릴 수 있음)" if warmup_steps <= 0 else f"warmup_steps={warmup_steps}"
    print(f"[diag] (4) 스트림/워밍업 : {wmsg}")
    print("============================================================")


# ------------------------
# Train / Eval with hard debug
# ------------------------
def run_one_epoch(model, optimizer, loader, device, train: bool, cfg, epoch: int,
                  amp_on: bool, sink=None):
    model.train(train)
    ce = nn.CrossEntropyLoss(label_smoothing=float(cfg["label_smoothing"]))
    grad_accum = int(cfg["grad_accum"])
    log_interval = int(cfg["log_interval"])
    channels_last = bool(cfg["channels_last"])

    use_prefetch = bool(cfg["use_prefetcher"]) and device.type in ("xpu", "cuda") and len(loader) > 0
    prefetcher = LocalPrefetcher(loader, device, channels_last) if use_prefetch else None

    dbg_timing = int(os.getenv("DBG_TIMING", "0"))
    dbg_max_steps = int(os.getenv("DBG_MAX_STEPS", "0"))  # 0 = unlimited
    early_ended = False

    total = 0
    loss_sum = 0.0
    acc_sum = 0.0
    t0 = time.time()
    steps = len(loader)
    step_count = 0
    step_time_sum = 0.0

    def print_dbg(step, tf0, tf1, tl1, tb0, tb1, ts0, ts1, step_t0, do_bwd):
        if not dbg_timing:
            return
        f_ms = (tf1 - tf0) * 1000.0
        l_ms = (tl1 - tf1) * 1000.0
        b_ms = (tb1 - tb0) * 1000.0 if do_bwd else 0.0
        s_ms = (ts1 - ts0) * 1000.0 if do_bwd else 0.0
        step_ms = (time.perf_counter() - step_t0) * 1000.0
        print(f"[dbg] step={step} fwd={f_ms:.1f}ms loss={l_ms:.1f}ms bwd={b_ms:.1f}ms step={s_ms:.1f}ms total={step_ms:.1f}ms")

    with profiler_context() as prof:
        if prefetcher is not None:
            x, y = prefetcher.next()
            step = 0
            while x is not None:
                step_t0 = time.perf_counter()
                step += 1

                tf0 = time.perf_counter()
                with amp_ctx(device.type, amp_on):
                    logits = model(x)
                tf1 = time.perf_counter()

                with amp_ctx(device.type, False):
                    ce_val = ce_with_ur_fallback(ce, logits, y, float(cfg["label_smoothing"]))
                tl1 = time.perf_counter()

                loss = ce_val / (grad_accum if train else 1)
                loss_scalar = to_host_scalar(ce_val)

                if train:
                    tb0 = time.perf_counter(); loss.backward(); tb1 = time.perf_counter()
                    if (step % grad_accum) == 0:
                        ts0 = time.perf_counter(); optimizer.step(); ts1 = time.perf_counter()
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        ts0 = ts1 = tb1
                else:
                    tb0 = tb1 = ts0 = ts1 = tf1

                bs = x.size(0); total += bs
                loss_sum += loss_scalar * bs
                acc_sum  += top1_cpu(logits, y) * bs

                if device.type == "xpu":
                    torch.xpu.synchronize()
                elif device.type == "cuda":
                    torch.cuda.synchronize()
                print_dbg(step, tf0, tf1, tl1, tb0, tb1, ts0, ts1, step_t0, do_bwd=train)
                if prof is not None:
                    prof.step()

                step_count += 1
                step_time_sum += (time.perf_counter() - step_t0) * 1000.0

                if dbg_max_steps and step >= dbg_max_steps:
                    early_ended = True
                    break

                if train and step % log_interval == 0:
                    elapsed = time.time() - t0
                    ips = total / max(elapsed, 1e-9)
                    avg_step = step_time_sum / max(step_count, 1)
                    print(
                        f"Epoch {epoch} [{step}/{steps}] loss={loss_sum/total:.4f} acc={acc_sum/total:.4f} "
                        f"throughput={ips:.1f} img/s avg_step={avg_step:.1f} ms elapsed={elapsed:.1f}s"
                    )

                x, y = prefetcher.next()

        else:
            memfmt = torch.channels_last if channels_last else torch.contiguous_format
            for step, (x, y) in enumerate(loader, 1):
                step_t0 = time.perf_counter()
                x = x.to(device, memory_format=memfmt, non_blocking=True)
                y = y.to(device, non_blocking=True)

                tf0 = time.perf_counter()
                with amp_ctx(device.type, amp_on):
                    logits = model(x)
                tf1 = time.perf_counter()

                with amp_ctx(device.type, False):
                    ce_val = ce_with_ur_fallback(ce, logits, y, float(cfg["label_smoothing"]))
                tl1 = time.perf_counter()

                loss = ce_val / (grad_accum if train else 1)
                loss_scalar = to_host_scalar(ce_val)

                if train:
                    tb0 = time.perf_counter(); loss.backward(); tb1 = time.perf_counter()
                    if (step % grad_accum) == 0:
                        ts0 = time.perf_counter(); optimizer.step(); ts1 = time.perf_counter()
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        ts0 = ts1 = tb1
                else:
                    tb0 = tb1 = ts0 = ts1 = tf1

                bs = x.size(0); total += bs
                loss_sum += loss_scalar * bs
                acc_sum  += top1_cpu(logits, y) * bs

                if device.type == "xpu":
                    torch.xpu.synchronize()
                elif device.type == "cuda":
                    torch.cuda.synchronize()
                print_dbg(step, tf0, tf1, tl1, tb0, tb1, ts0, ts1, step_t0, do_bwd=train)
                if prof is not None:
                    prof.step()

                step_count += 1
                step_time_sum += (time.perf_counter() - step_t0) * 1000.0

                if dbg_max_steps and step >= dbg_max_steps:
                    early_ended = True
                    break

                if train and (step % log_interval) == 0:
                    elapsed = time.time() - t0
                    ips = total / max(elapsed, 1e-9)
                    avg_step = step_time_sum / max(step_count, 1)
                    print(
                        f"Epoch {epoch} [{step}/{steps}] loss={loss_sum/total:.4f} acc={acc_sum/total:.4f} "
                        f"throughput={ips:.1f} img/s avg_step={avg_step:.1f} ms elapsed={elapsed:.1f}s"
                    )

    elapsed = time.time() - t0
    ips = total / max(elapsed, 1e-9)
    print(f"[epoch-summary] epoch={epoch} time={elapsed:.2f}s ips={ips:.1f} img/s early_end={early_ended}")
    return loss_sum / max(total, 1), acc_sum / max(total, 1), elapsed, early_ended


# ------------------------
# Main
# ------------------------
def main(args):
    # ----- bake test toggles into env BEFORE importing fused ops -----
    if args.dbg_timing != "auto":
        os.environ["DBG_TIMING"] = "1" if args.dbg_timing else "0"
    if args.dbg_max_steps is not None:
        os.environ["DBG_MAX_STEPS"] = str(int(args.dbg_max_steps))

    if args.xpu_fold_on_cpu != "auto":
        os.environ["XPU_FUSED_FOLD_ON_CPU"] = "1" if args.xpu_fold_on_cpu else "0"
    if args.xpu_kernel != "auto":
        # e.g. tiled | unset | ref (passed through to the C++ side)
        os.environ["XPU_FUSED_KERNEL"] = str(args.xpu_kernel)

    if args.omp_threads is not None and args.omp_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(int(args.omp_threads))

    # ----- banner & env echo -----
    dbg_timing = os.getenv("DBG_TIMING", "0")
    dbg_max = os.getenv("DBG_MAX_STEPS", "0")
    xpu_fold = os.getenv("XPU_FUSED_FOLD_ON_CPU", "auto")
    xpu_kernel = os.getenv("XPU_FUSED_KERNEL", "auto")
    print(f"[dbg-banner] DBG_TIMING={dbg_timing} DBG_MAX_STEPS={dbg_max}")
    print(f"[fused-banner] XPU_FUSED_FOLD_ON_CPU={xpu_fold} XPU_FUSED_KERNEL={xpu_kernel}")

    # import AFTER env wiring so the extension can see the env
    from models.ops_fused_sycl_train import ensure_fused_decided, build_model

    cfg = ensure_cfg_defaults(load_yaml(args.config))

    fused_ok = ensure_fused_decided(args.fused)
    os.environ["ENABLE_XPU_FUSED"] = "on" if fused_ok else "off"

    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "XPU device not available"
    device = torch.device("xpu")
    print(f"[device] Using device: {device.type}")

    dataset = cfg["dataset"]; dataloader = cfg["dataloader"]
    img_size = int(dataset["img_size"]); root = dataset["root"]
    download = bool(dataset["download"])
    mean = tuple(dataset.get("mean", [])) or None
    std  = tuple(dataset.get("std",  [])) or None

    tf_train = [
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    tf_eval  = [
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ]
    if mean and std:
        tf_train.append(transforms.Normalize(mean, std))
        tf_eval.append(transforms.Normalize(mean, std))
    tf_train = transforms.Compose(tf_train)
    tf_eval  = transforms.Compose(tf_eval)

    train_set = datasets.CIFAR10(root=root, train=True,  download=download, transform=tf_train)
    test_set  = datasets.CIFAR10(root=root, train=False, download=download, transform=tf_eval)

    bs = int(dataloader["batch_size"])          
    eval_bs = int(dataloader["eval_batch_size"]) 
    num_workers = int(dataloader["num_workers"]) 
    pin_memory = bool(dataloader["pin_memory"]) 
    prefetch_factor = int(dataloader["prefetch_factor"]) 
    eval_prefetch_factor = int(dataloader["eval_prefetch_factor"]) 
    persistent_workers = bool(dataloader["persistent_workers"]) 

    if args.safe_loader:
        num_workers, pin_memory, prefetch_factor, persistent_workers = 0, False, 2, False
        eval_prefetch_factor = 2
        print("[loader] XPU-SAFE: num_workers=0, pin_memory=False, prefetch_factor=2, persistent_workers=False")
    else:
        if args.num_workers is not None: num_workers = int(args.num_workers)
        if str(args.pin_memory).lower() != "auto": pin_memory = bool(args.pin_memory)
        if args.prefetch_factor is not None:
            prefetch_factor = int(args.prefetch_factor); eval_prefetch_factor = prefetch_factor
        if str(args.persistent_workers).lower() != "auto": persistent_workers = bool(args.persistent_workers)

    loader_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,  drop_last=False, **loader_kwargs)
    test_loader  = DataLoader(test_set,  batch_size=eval_bs, shuffle=False, drop_last=False, **loader_kwargs)

    model = build_model(cfg).to(device)
    ch_flag = args.channels_last
    channels_last_on = (cfg["misc"]["channels_last"] if ch_flag == "auto" else bool(ch_flag))
    if channels_last_on:
        model = set_channels_last(model)
    if args.diag_fused:
        emit_fused_diagnosis(channels_last_on, int(args.warmup_fused))
    if fused_ok and args.warmup_fused > 0:
        run_fused_warmup(int(args.warmup_fused))

    tr = cfg["train"]
    if tr.get("optimizer", "sgd").lower() == "sgd":
        opt = torch.optim.SGD(
            model.parameters(),
            lr=float(tr["lr"]),
            momentum=float(tr["momentum"]),
            weight_decay=float(tr["weight_decay"]),
        )
    else:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(tr["lr"]),
            weight_decay=float(tr["weight_decay"]),
        )

    if args.ipex == "on" and device.type == "xpu":
        try:
            import intel_extension_for_pytorch as ipex
            model, opt = ipex.optimize(model, optimizer=opt, level="O1", dtype=torch.float32, inplace=True)
            print("[ipex] applied optimize(level=O1, compute_dtype=fp32)")
        except Exception as e:
            print(f"[ipex] optimize failed: {e}")

    rt_cfg = dict(
        label_smoothing=float(tr["label_smoothing"]),
        grad_accum=int(tr["grad_accum"]),
        log_interval=int(cfg["misc"]["log_interval"]),
        channels_last=channels_last_on,
        use_prefetcher=bool(cfg["misc"]["use_prefetcher"]),
    )

    # ---- train (may early-exit) ----
    epochs = int(tr["epochs"]) 
    best = 0.0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_time, early = run_one_epoch(
            model, opt, train_loader, device, True, rt_cfg, epoch, amp_on=False
        )
        if early:
            print("[dbg] early-exit after train per DBG_MAX_STEPS — skipping eval & further epochs.")
            return
        ev_loss, ev_acc, ev_time, _ = run_one_epoch(
            model, opt, test_loader, device, False, rt_cfg, epoch, amp_on=False
        )
        print(
            f"[Epoch {epoch}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} time={tr_time:.1f}s | "
            f"val_loss={ev_loss:.4f} val_acc={ev_acc:.4f} time={ev_time:.1f}s"
        )
        if ev_acc > best:
            best = ev_acc
            ckpt_path = os.path.join(cfg["misc"]["ckpt_dir"], cfg["misc"]["ckpt_name"])
            save_checkpoint(ckpt_path, model, epoch, best, extra={"val_acc": best})
            print(f"  ↳ Saved best checkpoint @ {ckpt_path} (val_acc={best:.4f})")


# ------------------------
# CLI
# ------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML config path")

    # Fused op control
    p.add_argument("--fused", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--xpu-fold-on-cpu", type=str2bool, default="auto",
                   help="Fold BN on CPU once (on/off/auto)")
    p.add_argument("--xpu-kernel", choices=["auto", "tiled", 'global', "unset", "ref"], default="auto",
                   help="Preferred fused kernel variant (passed via env)")
    p.add_argument("--diag-fused", action="store_true",
                   help="Print fused 커널 체커 (메모리/타일/워밍업) 요약")
    p.add_argument("--warmup-fused", type=int, default=0,
                   help="훈련 전 fused probe(3x3 conv) 워밍업 반복 횟수")

    # Runtime toggles baked into code (no export needed)
    p.add_argument("--dbg-timing", type=str2bool, default="auto")
    p.add_argument("--dbg-max-steps", type=int, default=None)
    p.add_argument("--omp-threads", type=int, default=None)

    # Perf helpers
    p.add_argument("--ipex",  choices=["on", "off"], default="on")
    p.add_argument("--channels-last", type=str2bool, default="auto")
    p.add_argument("--safe-loader", action="store_true")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--pin-memory", type=str2bool, default="auto")
    p.add_argument("--prefetch-factor", type=int, default=None)
    p.add_argument("--persistent-workers", type=str2bool, default="auto")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Defer fused import until after env wiring in main()
    main(args)
