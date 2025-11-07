import os, time, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.common import load_yaml, ensure_dirs, set_seed, to_channels_last, save_checkpoint
from utils.metrics import accuracy
from utils.augment import rand_mixup, rand_cutmix, mix_criterion
from models import build_model
from data.cifar10 import get_loaders

def main(cfg_path):
    cfg = load_yaml(cfg_path)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_loaders(cfg, device.type)
    model = build_model(cfg).to(device)
    model = to_channels_last(model, cfg["misc"]["channels_last"])

    if cfg["train"]["optimizer"].lower() == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=cfg["train"]["lr"],
                              momentum=cfg["train"]["momentum"],
                              weight_decay=cfg["train"]["weight_decay"], nesterov=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                weight_decay=cfg["train"]["weight_decay"])

    sched = CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["misc"]["amp"])
    ce = nn.CrossEntropyLoss(label_smoothing=cfg["train"]["label_smoothing"])

    ensure_dirs(cfg["misc"]["ckpt_dir"], cfg["misc"]["log_dir"])
    best_acc = 0.0
    ckpt_path = os.path.join(cfg["misc"]["ckpt_dir"], cfg["misc"]["ckpt_name"])

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        # 학습 단계
        model.train()
        running_loss, running_acc, total = 0.0, 0.0, 0
        start = time.time()

        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            mixinfo = None
            if cfg["train"]["cutmix_alpha"] > 0:
                x, (ya, yb), lam = rand_cutmix(x, y, cfg["train"]["cutmix_alpha"])
                mixinfo = ((ya, yb), lam)
            elif cfg["train"]["mixup_alpha"] > 0:
                x, (ya, yb), lam = rand_mixup(x, y, cfg["train"]["mixup_alpha"])
                mixinfo = ((ya, yb), lam)

            with torch.cuda.amp.autocast(enabled=cfg["misc"]["amp"]):
                logits = model(x)
                loss = mix_criterion(ce, logits, y, mixinfo)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = x.size(0)
            running_loss += loss.item() * bs
            running_acc  += accuracy(logits.detach(), y) * bs
            total += bs

            if step % cfg["misc"]["log_interval"] == 0:
                print(f"Epoch {epoch} [{step}/{len(train_loader)}] "
                      f"loss={running_loss/total:.4f} acc={running_acc/total:.4f}")

        sched.step()

        # 평가 단계
        model.eval()
        val_loss, val_acc, val_total = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=cfg["misc"]["amp"]):
                    logits = model(x)
                    loss = ce(logits, y)
                bs = x.size(0)
                val_loss += loss.item() * bs
                val_acc  += accuracy(logits, y) * bs
                val_total += bs

        val_loss /= val_total; val_acc /= val_total
        dur = time.time() - start
        print(f"[Epoch {epoch}] train_loss={running_loss/total:.4f} "
              f"train_acc={running_acc/total:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f} time={dur:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(ckpt_path, model, epoch, best_acc)
            print(f"  ↳ Saved best checkpoint @ {ckpt_path} (val_acc={best_acc:.4f})")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="./configs/cifar10.yaml")
    args = ap.parse_args()
    main(args.config)
