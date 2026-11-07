"""Classify a weed image with a trained WeedNet-w2 checkpoint and save its attention map.

Usage:
    python predict.py --weights weights/weednet_w2_seed42.pt --image path/to/photo.jpg [--save-map map.png]
"""
import argparse

import torch
import torch.nn.functional as F
from PIL import Image

from weednet import WeedNetW2, SPECIES, NUM_CLASSES, get_eval_transform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="state_dict saved from the thesis notebook (<id>.pt)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--save-map", default=None, help="optional path for an attention-map overlay (PNG)")
    args = ap.parse_args()

    model = WeedNetW2(NUM_CLASSES, attention_head=True, pretrained_stem=False)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    img = Image.open(args.image).convert("RGB")
    x = get_eval_transform()(img).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    top = probs.topk(args.topk)
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"{SPECIES[i]:<30s} {p * 100:5.1f}%")

    if args.save_map:
        import matplotlib.pyplot as plt
        amap = model.attention_map()[0]
        amap = F.interpolate(amap[None, None], size=img.size[::-1], mode="bilinear", align_corners=False)[0, 0]
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.imshow(amap.numpy(), cmap="jet", alpha=0.45)
        plt.title(f"{SPECIES[top.indices[0]]} ({top.values[0] * 100:.1f}%)")
        plt.axis("off")
        plt.savefig(args.save_map, bbox_inches="tight", dpi=150)
        print(f"Attention map saved to {args.save_map}")


if __name__ == "__main__":
    main()
