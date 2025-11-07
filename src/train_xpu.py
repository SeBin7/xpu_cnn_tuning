from __future__ import annotations
import os
import time
import json
import argparse
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.optim as optim

# --- 프로젝트 util 및 모델 모듈 ---
from utils.common import load_yaml, save_checkpoint
from utils.runtime import (
    build_cifar10_loaders,
    configure_cpu_threads,
    configure_device,
    pick_amp_dtype_avoid_bf16,
    prepare_environment,
    resolve_channels_last,
    str2bool,
)
from models import build_model

def _amp_ctx(device_type: str, amp_on: bool, amp_dtype: torch.dtype | None):
    if not amp_on or amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device_type, dtype=amp_dtype)

def _maybe_import_ipex(ipex_flag: str):
    if ipex_flag == "off":
        print("[ipex] disabled by flag (--ipex off)")
        return None
    try:
        import intel_extension_for_pytorch as ipex
        return ipex
    except Exception as e:
        if ipex_flag == "on":
            print(f"[ipex] requested but unavailable -> continue without IPEX. reason: {e}")
        elif ipex_flag == "auto":
            print(f"[ipex] auto: unavailable or failed -> continue without IPEX. reason: {e}")
        return None

def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()

def mix_criterion(ce: nn.Module, logits, targets, mixinfo):
    return ce(logits, targets)

def run_one_epoch(model, optimizer, scheduler, loaders, cfg, device, amp_ctx, scaler, epoch, train=True, sink=None):
    # 토글 설정
    do_prof = bool(cfg["misc"].get("profile_step_timing", False))
    log_interval = int(cfg["misc"].get("log_interval", 50))
    channels_last_on = bool(cfg["misc"].get("channels_last", False))
    mem_format = torch.channels_last if channels_last_on else torch.contiguous_format

    if train:
        model.train()
        data_loader = loaders[0]
    else:
        model.eval()
        data_loader = loaders[1]

    # 손실 함수
    ls = float(cfg["train"].get("label_smoothing", 0.0))
    ce = nn.CrossEntropyLoss(label_smoothing=ls)
    grad_accum = int(cfg["train"].get("grad_accum", 1))

    total = 0
    loss_sum = 0.0
    acc_sum = 0.0
    t0 = time.time()
    step_count = 0
    step_time_sum = 0.0
    len_loader = len(data_loader)

    for step, (x, y) in enumerate(data_loader, 1):
        # H2D 복사: pin_memory 설정과 일관되게 non_blocking을 유지
        non_block = bool(cfg["data"].get("non_blocking", True))
        step_t0 = time.perf_counter()

        x = x.to(device=device, memory_format=mem_format, non_blocking=non_block)
        y = y.to(device=device, non_blocking=non_block)

        with amp_ctx():
            logits = model(x)
            loss = mix_criterion(ce, logits, y, None) / (grad_accum if train else 1)

        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step % grad_accum) == 0:
                if scaler is not None:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

        # 타이밍 안정화
        if device.type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        step_ms = (time.perf_counter() - step_t0) * 1000.0
        step_count += 1
        step_time_sum += step_ms

        bs = x.size(0)
        total += bs
        loss_sum += (loss.detach() * (grad_accum if train else 1)).item() * bs
        acc_sum += accuracy_top1(logits, y) * bs

        if step % log_interval == 0:
            elapsed = time.time() - t0
            ips = total / max(elapsed, 1e-6)
            avg_step = step_time_sum / max(step_count, 1)
            print(
                f"Epoch {epoch} [{step}/{len_loader}] "
                f"loss={loss_sum/total:.4f} acc={acc_sum/total:.4f} "
                f"throughput={ips:.1f} img/s "
                f"step={step_ms:.1f} ms avg_step={avg_step:.1f} ms "
                f"elapsed={elapsed:.1f}s"
            )
            if sink is not None:
                sink.write({
                    "phase": "train" if train else "eval",
                    "epoch": epoch, "step": step,
                    "loss": round(loss_sum/total, 6),
                    "acc": round(acc_sum/total, 6),
                    "throughput_img_s": round(ips, 2),
                    "step_ms": round(step_ms, 3),
                    "avg_step_ms": round(avg_step, 3),
                    "elapsed_s": round(elapsed, 2),
                    "device": device.type,
                    "amp": cfg["misc"].get("_amp_dtype_used", "fp32"),
                    "channels_last": bool(cfg["misc"].get("_channels_last_active", False)),
                })

    elapsed = time.time() - t0
    epoch_ips = total / max(elapsed, 1e-6)
    print(f"[epoch-summary] epoch={epoch} time={elapsed:.2f}s ips={epoch_ips:.1f} img/s")
    return loss_sum / max(total, 1), acc_sum / max(total, 1), elapsed

def main(args):
    cfg = load_yaml(args.config)

    misc = prepare_environment(
        cfg,
        default_channels_last=False,
        default_use_prefetcher=False,
        default_log_interval=50,
    )
    device = configure_device(misc)
    configure_cpu_threads(misc)

    data_cfg = cfg.setdefault("data", {})
    data_cfg["non_blocking"] = bool(data_cfg.get("non_blocking", True))

    loaders = build_cifar10_loaders(cfg, args, device, misc)
    train_loader, test_loader = loaders.train, loaders.eval
    num_classes = loaders.num_classes

    # ---- 모델 ----
    name = cfg["model"]["name"]
    cfg["model"]["num_classes"] = int(cfg["model"].get("num_classes", num_classes))
    model = build_model(name, cfg).to(device)

    # channels_last 적용
    channels_last_on = resolve_channels_last(cfg["misc"], args.channels_last)
    cfg["misc"]["channels_last"] = channels_last_on
    cfg["misc"]["_channels_last_active"] = channels_last_on
    if channels_last_on:
        model = model.to(memory_format=torch.channels_last)

    # ---- 옵티마이저 / 스케줄러 ----
    opt_name = (cfg["train"].get("optimizer", "sgd")).lower()
    lr = float(cfg["train"].get("lr", 0.002))
    momentum = float(cfg["train"].get("momentum", 0.9))
    weight_decay = float(cfg["train"].get("weight_decay", 1e-4))
    if opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")
    scheduler = None

    # ---- IPEX(FP32 전용, BF16 금지) ----
    ipex = _maybe_import_ipex(args.ipex)
    if ipex is not None and device.type == "xpu":
        try:
            model, optimizer = ipex.optimize(
                model, optimizer=optimizer, level="O1", dtype=torch.float32, inplace=True
            )
            print("[ipex] applied optimize(level=O1, weights=fp32, compute_dtype=fp32)")
        except Exception as e:
            print(f"[ipex] optimize failed -> continue without IPEX. reason: {e}")

    # ---- AMP(bf16 회피) ----
    amp_on = (args.amp == "on")
    req = args.amp_dtype
    if req == "bf16":
        print("[amp] bf16 is disallowed by policy -> falling back to fp32")
        req = "auto"
    amp_dtype, used_amp_name = pick_amp_dtype_avoid_bf16(device.type, req if amp_on else "off")
    cfg["misc"]["_amp_dtype_used"] = used_amp_name
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and amp_on and amp_dtype==torch.float16)) if amp_on else None

    # ---- JSONL 로거 ----
    sink = None
    if args.log_json:
        os.makedirs(os.path.dirname(args.log_json), exist_ok=True)
        class _JsonSink:
            def __init__(self, path): self.f = open(path, "a", buffering=1)
            def write(self, obj): self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            def close(self): self.f.close()
        sink = _JsonSink(args.log_json)
        print(f"[log] JSONL sink: {args.log_json}")

    # ---- 학습 / 평가 ----
    epochs = int(cfg["train"].get("epochs", 10))
    best = -1e9
    ckpt_path = os.path.join(cfg["misc"]["ckpt_dir"], cfg["misc"]["ckpt_name"])

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_time = run_one_epoch(
            model, optimizer, scheduler,
            (train_loader, test_loader), cfg, device, lambda: _amp_ctx(device.type, amp_on and amp_dtype is not None, amp_dtype), scaler, epoch, train=True, sink=sink
        )
        with torch.no_grad():
            ev_loss, ev_acc, ev_time = run_one_epoch(
                model, optimizer, scheduler,
                (train_loader, test_loader), cfg, device, lambda: nullcontext(), None, epoch, train=False, sink=sink
            )
        print(f"[Epoch {epoch}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} time={tr_time:.1f}s | "
              f"val_loss={ev_loss:.4f} val_acc={ev_acc:.4f} time={ev_time:.1f}s")

        if ev_acc > best:
            best = ev_acc
            save_checkpoint(ckpt_path, model, epoch, best, extra={"val_acc": best})
            print(f"  -> Saved best checkpoint @ {ckpt_path} (val_acc={best:.4f})")

    if sink is not None:
        sink.close()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--ipex", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--model", default=None, help="override model name (optional)")
    p.add_argument("--log-json", default=None, help="write JSONL metrics to this path")
    p.add_argument("--amp", choices=["on", "off"], default="off")
    p.add_argument("--amp-dtype", choices=["bf16", "fp16", "auto"], default="auto")
    p.add_argument("--channels-last", type=str2bool, default="auto")
    p.add_argument("--safe-loader", action="store_true", help="Set num_workers=0 & pin_memory=False quickly")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
