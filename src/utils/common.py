import os
import random
import yaml
import torch
import torchvision.transforms as T

DEFAULT_MEAN = (0.4914, 0.4822, 0.4465)
DEFAULT_STD = (0.2470, 0.2435, 0.2616)

def build_transforms(
    resize: int = 224,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
    augment: dict | None = None,
):
    """필요하면 증강이 포함된 (train_tf, test_tf) torchvision 변환을 반환합니다."""
    augment = augment or {}

    train_tfs: list = []
    # 리사이즈 전에 랜덤 크롭(선택적 패딩)을 적용하면 CIFAR 스타일 증강을 유지
    if augment.get("random_crop"):
        crop_size = int(augment.get("crop_size", resize))
        padding = int(augment.get("random_crop_pad", 4))
        pad_mode = augment.get("random_crop_pad_mode", "reflect")
        train_tfs.append(
            T.RandomCrop(crop_size, padding=padding, padding_mode=pad_mode)
        )
        if crop_size != resize:
            train_tfs.append(
                T.Resize((resize, resize), interpolation=T.InterpolationMode.BILINEAR)
            )
    else:
        train_tfs.append(
            T.Resize((resize, resize), interpolation=T.InterpolationMode.BILINEAR)
        )

    if augment.get("random_flip", True):
        train_tfs.append(T.RandomHorizontalFlip())

    color_jitter_cfg = augment.get("color_jitter")
    if isinstance(color_jitter_cfg, dict) and color_jitter_cfg:
        train_tfs.append(T.ColorJitter(**color_jitter_cfg))

    train_tfs.extend([T.ToTensor(), T.Normalize(mean, std)])

    cutout = augment.get("cutout", 0)
    if isinstance(cutout, (int, float)) and cutout > 0:
        train_tfs.append(
            T.RandomErasing(
                p=min(1.0, float(cutout)),
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
                inplace=True,
            )
        )

    test_tfs = [
        T.Resize((resize, resize), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean, std),
    ]

    return T.Compose(train_tfs), T.Compose(test_tfs)

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    # 사용 가능하다면 CUDA/XPU도 동일한 시드로 설정
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_channels_last(model: torch.nn.Module, enabled: bool = True):
    if enabled:
        model = model.to(memory_format=torch.channels_last)
    return model

def save_checkpoint(path: str, model: torch.nn.Module, epoch: int, best: float, extra: dict | None = None):
    obj = {"model": model.state_dict(), "epoch": epoch, "best": best}
    if extra:
        obj.update(extra)
    torch.save(obj, path)

def load_checkpoint(path: str, model: torch.nn.Module, map_location="cpu"):
    obj = torch.load(path, map_location=map_location)
    model.load_state_dict(obj["model"], strict=True)
    return obj
