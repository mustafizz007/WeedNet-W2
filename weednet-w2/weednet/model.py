"""WeedNet-w2: a compact attention-equipped CNN for weed species classification.

Architecture (541,506 parameters, 0.049 GMACs for a 3x224x224 image, 26 classes):

    input 3x224x224
      imagenet_stem  MobileNetV3-Small features[0:9], ImageNet weights, trainable   190,520 params -> 48x14x14
      domain_blocks  3 MobileNetV3-style inverted-residual blocks (SE, h-swish),
                     randomly initialised, + 1x1 conv to 320 channels               233,456 params -> 320x7x7
      head           QKV attention head (W2, proposed)                               117,530 params
                       1x1 conv 320->128 -> 49 tokens + learnable [CLS] token + position embedding
                       LayerNorm -> Q, K, V (4 heads x 32) -> softmax(QK^T / sqrt(32)) V -> residual
                       LayerNorm -> Dropout 0.2 -> Linear(128 -> num_classes) on the [CLS] token
                     or plain pooled head (W1, ablation): GAP -> FC 320->256 -> h-swish -> Dropout -> FC

The [CLS] token's attention over the 7x7 positions is stored in ``model.head.last_attn`` after every forward
pass and can be used as a built-in explanation map (see ``attention_map``).

Extracted from the thesis notebook (``notebooks/weednet-final.ipynb``, where the class is called ``WeedNetV4``).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

IMG_SIZE = 224
STEM_LAYERS = 9          # MobileNetV3-Small features[0:9]
ATTN_DIM = 128
ATTN_HEADS = 4
LAST_CONV_CH = 320
HEAD_HIDDEN = 256        # plain head (W1) only

DOMAIN_BLOCK_SPEC = [    # (expand channels, out channels, kernel, stride, squeeze-and-excitation)
    (192, 80, 5, 2, True),
    (240, 80, 5, 1, True),
    (240, 96, 3, 1, True),
]


def _make_divisible(v, divisor=8):
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    return new_v + divisor if new_v < 0.9 * v else new_v


class SqueezeExcite(nn.Module):
    """Channel attention: global pool -> bottleneck -> hard-sigmoid gate per channel."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        squeeze = _make_divisible(channels // reduction)
        self.fc1 = nn.Conv2d(channels, squeeze, 1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze, channels, 1)
        self.gate = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        return x * self.gate(self.fc2(self.act(self.fc1(s))))


class InvertedResidualV3(nn.Module):
    """MobileNetV3 bottleneck: 1x1 expand -> kxk depthwise -> (SE) -> 1x1 project, residual when shapes match."""

    def __init__(self, in_ch, exp_ch, out_ch, kernel, stride, use_se=True):
        super().__init__()
        layers = []
        if exp_ch != in_ch:
            layers += [nn.Conv2d(in_ch, exp_ch, 1, bias=False), nn.BatchNorm2d(exp_ch), nn.Hardswish(inplace=True)]
        layers += [nn.Conv2d(exp_ch, exp_ch, kernel, stride, kernel // 2, groups=exp_ch, bias=False),
                   nn.BatchNorm2d(exp_ch), nn.Hardswish(inplace=True)]
        if use_se:
            layers.append(SqueezeExcite(exp_ch))
        layers += [nn.Conv2d(exp_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)]
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_ch == out_ch

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_residual else y


class AttentionHead(nn.Module):
    """Multi-head self-attention (Q/K/V) over the HxW feature map plus a learnable [CLS] token."""

    def __init__(self, in_ch, num_classes, hw, dim=ATTN_DIM, heads=ATTN_HEADS, dropout=0.2):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.d = heads, dim // heads
        self.reduce = nn.Conv2d(in_ch, dim, 1)                          # 320 -> 128 channels = token size
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))                  # learnable summary token
        self.pos = nn.Parameter(torch.zeros(1, hw[0] * hw[1] + 1, dim))  # learnable position embedding
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)                               # Query, Key and Value in one layer
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, num_classes)
        self.last_attn = None                                            # (batch, heads, tokens, tokens)

    def forward(self, x):
        B = x.shape[0]
        t = self.reduce(x).flatten(2).transpose(1, 2)                    # (B, 49, dim): one token per position
        t = torch.cat([self.cls.expand(B, -1, -1), t], dim=1) + self.pos
        q, k, v = self.qkv(self.norm1(t)).reshape(B, -1, 3, self.heads, self.d).permute(2, 0, 3, 1, 4)
        attn = ((q @ k.transpose(-2, -1)) / self.d ** 0.5).softmax(dim=-1)  # (B, heads, 50, 50)
        self.last_attn = attn.detach()
        t = t + self.proj((attn @ v).transpose(1, 2).reshape(B, -1, self.heads * self.d))
        return self.fc(self.drop(self.norm2(t[:, 0])))                   # classify the [CLS] token


class WeedNetW2(nn.Module):
    """WeedNet-w2 (``attention_head=True``) or its plain-head ablation W1 (``attention_head=False``).

    Args:
        num_classes: number of output classes (26 in the thesis).
        attention_head: use the QKV attention head (W2) instead of the pooled MLP head (W1).
        pretrained_stem: load ImageNet weights into the MobileNetV3-Small stem. Set to False when you are
            about to load a trained WeedNet-w2 checkpoint anyway (avoids downloading the ImageNet weights).
    """

    def __init__(self, num_classes=26, attention_head=True, pretrained_stem=True):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained_stem else None
        donor = models.mobilenet_v3_small(weights=weights)
        self.imagenet_stem = nn.Sequential(*list(donor.features.children())[:STEM_LAYERS])

        with torch.no_grad():
            ch = self.imagenet_stem.eval()(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)).shape[1]
        blocks = []
        for exp_ch, out_ch, k, s, se in DOMAIN_BLOCK_SPEC:
            blocks.append(InvertedResidualV3(ch, exp_ch, out_ch, k, s, se))
            ch = out_ch
        blocks.append(nn.Sequential(nn.Conv2d(ch, LAST_CONV_CH, 1, bias=False),
                                    nn.BatchNorm2d(LAST_CONV_CH), nn.Hardswish(inplace=True)))
        self.domain_blocks = nn.Sequential(*blocks)

        if attention_head:
            with torch.no_grad():
                feat = self.domain_blocks.eval()(self.imagenet_stem(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)))
            self.head = AttentionHead(LAST_CONV_CH, num_classes, tuple(feat.shape[-2:]))
        else:
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                      nn.Linear(LAST_CONV_CH, HEAD_HIDDEN), nn.Hardswish(inplace=True),
                                      nn.Dropout(0.2), nn.Linear(HEAD_HIDDEN, num_classes))
        self._init_random(self.domain_blocks)
        self._init_random(self.head)
        self.train()

    @staticmethod
    def _init_random(module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
            elif isinstance(m, AttentionHead):
                nn.init.trunc_normal_(m.cls, std=0.02)
                nn.init.trunc_normal_(m.pos, std=0.02)

    def forward(self, x):
        return self.head(self.domain_blocks(self.imagenet_stem(x)))

    def attention_map(self):
        """[CLS] attention over the 7x7 positions from the last forward pass, averaged over heads: (B, 7, 7)."""
        if not isinstance(self.head, AttentionHead) or self.head.last_attn is None:
            raise RuntimeError("Run a forward pass with the attention head (W2) first.")
        a = self.head.last_attn.mean(dim=1)[:, 0, 1:]                     # [CLS] row, without itself
        side = int(a.shape[-1] ** 0.5)
        return a.reshape(-1, side, side)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def count_macs(model, img_size=IMG_SIZE):
    """Multiply-accumulates for one image (convolutions + linear layers; attention products excluded), in G."""
    macs = [0]

    def conv_hook(m, inp, out):
        macs[0] += (out.numel() // out.shape[0]) * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def linear_hook(m, inp, out):
        macs[0] += (out.numel() // out.shape[-1]) // out.shape[0] * m.in_features * m.out_features

    hooks = [m.register_forward_hook(conv_hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    hooks += [m.register_forward_hook(linear_hook) for m in model.modules() if isinstance(m, nn.Linear)]
    was_training = model.training
    model.eval()
    model(torch.zeros(1, 3, img_size, img_size, device=next(model.parameters()).device))
    for h in hooks:
        h.remove()
    model.train(was_training)
    return macs[0] / 1e9


if __name__ == "__main__":
    for label, attn in (("WeedNet-w2 (W2, QKV attention head)", True), ("Plain-head ablation (W1)", False)):
        m = WeedNetW2(26, attention_head=attn, pretrained_stem=False)
        parts = {n: count_params(getattr(m, n)) for n in ("imagenet_stem", "domain_blocks", "head")}
        print(f"{label}: {count_params(m):,} params, {count_macs(m):.3f} GMACs  {parts}")
