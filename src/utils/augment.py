import torch
import numpy as np

def rand_mixup(x, y, alpha=0.2):
    if alpha <= 0: return x, y, None
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, (y, y[idx]), lam

def rand_cutmix(x, y, alpha=1.0):
    if alpha <= 0: return x, y, None
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = x.size()
    idx = torch.randperm(B, device=x.device)

    cx, cy = np.random.randint(W), np.random.randint(H)
    w = int(W * (1 - lam) ** 0.5)
    h = int(H * (1 - lam) ** 0.5)
    x1, y1 = max(cx - w // 2, 0), max(cy - h // 2, 0)
    x2, y2 = min(cx + w // 2, W), min(cy + h // 2, H)

    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x, (y, y[idx]), lam

def mix_criterion(ce, logits, target, mixinfo):
    if mixinfo is None:
        return ce(logits, target)
    (y_a, y_b), lam = mixinfo
    return lam * ce(logits, y_a) + (1 - lam) * ce(logits, y_b)
