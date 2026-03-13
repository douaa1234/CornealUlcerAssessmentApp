import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

#image size
SAVE_SIZE = 512

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UNetResNet34(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        base= models.resnet34(weights=None)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu)  # /2
        self.pool = base.maxpool                                    # /4
        self.enc1 = base.layer1                                     # /4
        self.enc2 = base.layer2                                     # /8
        self.enc3 = base.layer3                                     # /16
        self.enc4 = base.layer4                                     # /32

        self.center = ConvBNReLU(512, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = ConvBNReLU(256 + 256, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = ConvBNReLU(128 + 128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = ConvBNReLU(64 + 64, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.dec1 = ConvBNReLU(64 + 64, 64)

        self.up0 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec0 = ConvBNReLU(32, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.enc1(self.pool(x0))
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        c = self.center(x4)

        u4 = self.up4(c);  d4 = self.dec4(torch.cat([u4, x3], 1))
        u3 = self.up3(d4); d3 = self.dec3(torch.cat([u3, x2], 1))
        u2 = self.up2(d3); d2 = self.dec2(torch.cat([u2, x1], 1))
        u1 = self.up1(d2); d1 = self.dec1(torch.cat([u1, x0], 1))
        u0 = self.up0(d1); d0 = self.dec0(u0)
        return self.out(d0)


def read_rgb_u8(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)
    return rgb


def to_tensor_imagenet(rgb_u8: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(rgb_u8).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = (x - mean) / std
    return x.unsqueeze(0)


def load_ulcer_unet(ckpt_path: str, device: str | None = None):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = UNetResNet34(pretrained=False).to(dev)
    ckpt = torch.load(ckpt_path, map_location=dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, dev


@torch.no_grad()
def predict_mask_from_path(model, device, image_path: str, thr: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    #Returns: (rgb_u8_512, mask01_512)
    rgb= read_rgb_u8(image_path)
    x = to_tensor_imagenet(rgb).to(device)
    logits = model(x)[0, 0].cpu().numpy()
    prob = 1.0 / (1.0 + np.exp(-logits))
    mask01 = (prob >= float(thr)).astype(np.uint8)
    return rgb, mask01
