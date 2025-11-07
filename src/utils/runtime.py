# 파일: src/utils/runtime.py
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from utils.common import ensure_dirs, set_seed, build_transforms

_TRUE_STRINGS = {"on", "true", "1", "yes", "y"}
_FALSE_STRINGS = {"off", "false", "0", "no", "n"}
_AUTO_STRINGS = {"auto"}


def str2bool(value: str, *, allow_auto: bool = True) -> bool | str:
    """argparse에서 사용하기 좋은 불리언 파서로 필요하면 'auto'를 반환합니다."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if allow_auto and normalized in _AUTO_STRINGS:
        return "auto"
    if normalized in _TRUE_STRINGS:
        return True
    if normalized in _FALSE_STRINGS:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value}")


def prepare_environment(
    cfg: Dict[str, Any],
    *,
    default_channels_last: bool = False,
    default_use_prefetcher: bool = True,
    default_log_interval: int = 10,
) -> Dict[str, Any]:
    """기본 설정을 적용하고 디렉터리를 만들며 난수 시드를 고정합니다."""
    misc = cfg.setdefault("misc", {})
    misc.setdefault("ckpt_dir", "./outputs/ckpts")
    misc.setdefault("ckpt_name", f"{cfg['model']['name']}_best.pth")
    misc.setdefault("log_dir", "./outputs/logs")
    misc.setdefault("log_interval", default_log_interval)
    misc.setdefault("channels_last", default_channels_last)
    misc.setdefault("use_prefetcher", default_use_prefetcher)
    misc.setdefault("ze_affinity_mask", "auto")
    ensure_dirs(misc["ckpt_dir"], misc["log_dir"])

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    return misc


def configure_device(misc: Dict[str, Any]) -> torch.device:
    """ZE_AFFINITY 설정을 반영하여 디바이스를 선택하고 사용자가 확인할 수 있게 출력합니다."""
    ze_mask = str(misc.get("ze_affinity_mask", "auto"))
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        if ze_mask not in ("auto", "", None):
            prev = os.environ.get("ZE_AFFINITY_MASK")
            if prev != ze_mask:
                os.environ["ZE_AFFINITY_MASK"] = ze_mask
                print(f"[env] ZE_AFFINITY_MASK={ze_mask} (was {prev})")
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[device] Using device: {device.type}")
    return device


def configure_cpu_threads(misc: Dict[str, Any]) -> None:
    """필요에 따라 intra-op, inter-op CPU 스레드 제한을 적용합니다."""
    cpu_threads = int(misc.get("cpu_threads", 0))
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
    interop_threads = int(misc.get("cpu_interop_threads", 0))
    if interop_threads > 0 and hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(interop_threads)


@dataclass
class LoaderBundle:
    train: DataLoader
    eval: DataLoader
    prefetch_factor: int
    eval_prefetch_factor: int
    pin_memory: bool
    pin_memory_device: Optional[str]
    num_workers: int
    num_classes: int


def _is_auto(value: Any) -> bool:
    return isinstance(value, str) and value.lower() == "auto"


def build_cifar10_loaders(
    cfg: Dict[str, Any],
    args: Any,
    device: torch.device,
    misc: Dict[str, Any],
) -> LoaderBundle:
    """설정과 CLI 재정의를 반영하여 CIFAR-10 학습/평가용 데이터로더를 생성합니다."""
    data = cfg.setdefault("data", {})
    dataset_cfg = cfg.get("dataset", {})
    dataloader_cfg = cfg.get("dataloader", {})

    dataset_name = dataset_cfg.get("name", data.get("name", "cifar10")).lower()
    if dataset_name != "cifar10":
        raise ValueError(f"Unsupported dataset: {dataset_name}. Only 'cifar10' is wired in.")

    root = dataset_cfg.get("root", data.get("root", "./data"))
    resize = int(dataset_cfg.get("img_size", data.get("img_size", 32)))
    augment_cfg = dataset_cfg.get("augment") or data.get("augment") or {}
    download = bool(dataset_cfg.get("download", data.get("download", True)))

    build_kwargs = dict(resize=resize, augment=augment_cfg)
    mean = dataset_cfg.get("mean", data.get("mean"))
    std = dataset_cfg.get("std", data.get("std"))
    if mean is not None:
        build_kwargs["mean"] = tuple(mean)
    if std is not None:
        build_kwargs["std"] = tuple(std)
    train_tf, test_tf = build_transforms(**build_kwargs)

    batch_size = int(dataloader_cfg.get("batch_size", data.get("batch_size", 32)))
    eval_batch_size = int(dataloader_cfg.get("eval_batch_size", batch_size))
    num_workers = int(dataloader_cfg.get("num_workers", data.get("num_workers", 8)))
    pin_memory = bool(dataloader_cfg.get("pin_memory", data.get("pin_memory", True)))
    prefetch_factor = int(dataloader_cfg.get("prefetch_factor", data.get("prefetch_factor", 4)))
    eval_prefetch_factor = int(dataloader_cfg.get("eval_prefetch_factor", prefetch_factor))
    persistent_workers = bool(dataloader_cfg.get("persistent_workers", data.get("persistent_workers", True)))
    drop_last = bool(dataloader_cfg.get("drop_last", False))
    shuffle = bool(dataloader_cfg.get("shuffle", True))
    eval_shuffle = bool(dataloader_cfg.get("eval_shuffle", False))
    pin_memory_device = dataloader_cfg.get("pin_memory_device", misc.get("pin_memory_device", "auto"))

    if getattr(args, "safe_loader", False):
        num_workers, pin_memory, prefetch_factor, persistent_workers = 0, False, 2, False
        eval_prefetch_factor = prefetch_factor
        print("[loader] SAFE mode: num_workers=0, pin_memory=False, prefetch_factor=2, persistent_workers=False")
    else:
        if getattr(args, "num_workers", None) is not None:
            num_workers = int(args.num_workers)
        if not _is_auto(getattr(args, "pin_memory", "auto")):
            pin_memory = bool(args.pin_memory)
        if getattr(args, "prefetch_factor", None) is not None:
            prefetch_factor = int(args.prefetch_factor)
            eval_prefetch_factor = prefetch_factor
        if not _is_auto(getattr(args, "persistent_workers", "auto")):
            persistent_workers = bool(args.persistent_workers)

    train_set = CIFAR10(root=root, train=True, download=download, transform=train_tf)
    eval_set = CIFAR10(root=root, train=False, download=download, transform=test_tf)

    loader_kwargs_train: Dict[str, Any] = {"num_workers": num_workers, "pin_memory": pin_memory}
    loader_kwargs_eval: Dict[str, Any] = {"num_workers": num_workers, "pin_memory": pin_memory}

    pin_memory_device_supported = "pin_memory_device" in DataLoader.__init__.__code__.co_varnames
    pin_dev = pin_memory_device
    if isinstance(pin_dev, str):
        pin_dev = pin_dev.lower()
    if pin_dev == "auto":
        pin_dev = device.type if device.type in ("cuda", "xpu") else None
    if pin_memory and pin_dev and pin_memory_device_supported:
        loader_kwargs_train["pin_memory_device"] = pin_dev
        loader_kwargs_eval["pin_memory_device"] = pin_dev

    if num_workers > 0:
        loader_kwargs_train["prefetch_factor"] = prefetch_factor
        loader_kwargs_train["persistent_workers"] = persistent_workers
        loader_kwargs_eval["prefetch_factor"] = eval_prefetch_factor
        loader_kwargs_eval["persistent_workers"] = persistent_workers

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        **loader_kwargs_train,
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=eval_batch_size,
        shuffle=eval_shuffle,
        drop_last=False,
        **loader_kwargs_eval,
    )

    return LoaderBundle(
        train=train_loader,
        eval=eval_loader,
        prefetch_factor=prefetch_factor,
        eval_prefetch_factor=eval_prefetch_factor,
        pin_memory=pin_memory,
        pin_memory_device=pin_dev if isinstance(pin_dev, str) else pin_dev,
        num_workers=num_workers,
        num_classes=10,
    )


def resolve_channels_last(misc: Dict[str, Any], flag: bool | str) -> bool:
    """misc 기본값과 CLI 플래그를 바탕으로 channels_last 적용 여부를 결정합니다."""
    if isinstance(flag, str) and flag.lower() == "auto":
        return bool(misc.get("channels_last", False))
    return bool(flag)


def pick_amp_dtype_avoid_bf16(device_type: str, requested: str) -> Tuple[Optional[torch.dtype], str]:
    """XPU에서는 BF16을 피하고 CUDA 지원 여부를 확인하는 공통 AMP 정책 헬퍼 함수입니다."""
    if device_type == "xpu":
        return None, "fp32"
    if device_type == "cuda":
        if requested in ("off", None, False):
            return None, "fp32"
        if requested in ("auto", "fp16"):
            try:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pass
                return torch.float16, "fp16"
            except Exception:
                return None, "fp32"
        return None, "fp32"
    return None, "fp32"


def build_runtime_loop_cfg(cfg: Dict[str, Any], misc: Dict[str, Any], channels_last: bool) -> Dict[str, Any]:
    """학습 루프에서 사용하는 런타임 설정 딕셔너리를 구성합니다."""
    train_cfg = cfg.setdefault("train", {})
    return {
        "label_smoothing": float(train_cfg.get("label_smoothing", 0.0)),
        "grad_accum": int(train_cfg.get("grad_accum", 1)),
        "log_interval": int(misc.get("log_interval", 10)),
        "channels_last": bool(channels_last),
        "use_prefetcher": bool(misc.get("use_prefetcher", True)),
    }
