import torch

@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    정확도를 [0,1] 범위로 계산합니다. 간헐적으로 발생하는 XPU 감소 문제를 피하기 위해
    작은 텐서는 CPU로 옮겨 비교 및 합산을 수행합니다.
    """
    # 1) 디바이스에서 argmax 수행(부담이 작음)
    pred = logits.argmax(dim=1)

    # 2) 매우 작은 텐서를 CPU로 옮겨 비교 및 합산
    pred_cpu = pred.detach().to("cpu")
    y_cpu = y.detach().to("cpu")

    correct = (pred_cpu == y_cpu).sum().item()
    return correct / y_cpu.size(0)
