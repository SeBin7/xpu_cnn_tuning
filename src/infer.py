import os, torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from utils.common import load_yaml, load_checkpoint
from models.tinycifarnet import build_model
from data.cifar10 import CIFAR10_MEAN, CIFAR10_STD, get_loaders

CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def build_transform():
    return transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

def predict_image(model, img_path, device):
    model.eval()
    tfm = build_transform()
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = F.softmax(logits, dim=1)[0]
        conf, idx = torch.max(prob, dim=0)
    return CLASSES[idx.item()], conf.item()

def main(cfg_path, ckpt_path, img_path=None):
    cfg = load_yaml(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    load_checkpoint(ckpt_path, model, map_location=device)

    if img_path:
        label, conf = predict_image(model, img_path, device)
        print(f"{os.path.basename(img_path)} -> {label} ({conf:.3f})")
    else:
        _, test_loader = get_loaders(cfg)
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += x.size(0)
        print(f"Test Acc: {correct/total:.4f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="./configs/cifar10.yaml")
    ap.add_argument("--ckpt", type=str, default="./outputs/checkpoints/best_cifar10.pt")
    ap.add_argument("--image", type=str, default=None)
    args = ap.parse_args()
    main(args.config, args.ckpt, args.image)
