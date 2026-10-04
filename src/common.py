"""Shared pieces used by training, evaluation and export."""
import cv2
import numpy as np
import timm
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

SIZE = 256
MEAN = (0.485, 0.456, 0.406)   # ImageNet statistics (the backbone was pretrained with them)
STD = (0.229, 0.224, 0.225)


def train_tf():
    return A.Compose([
        A.Resize(SIZE, SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
        A.GaussNoise(p=0.2),
        A.Normalize(MEAN, STD),
        ToTensorV2(),
    ])


def eval_tf():
    # No random augmentation for validation / test: metrics must be deterministic.
    return A.Compose([A.Resize(SIZE, SIZE), A.Normalize(MEAN, STD), ToTensorV2()])


class DefectDS(Dataset):
    def __init__(self, df, tf):
        self.paths = df.path.tolist()
        self.labels = df.label.tolist()
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)   # grayscale JPEG -> 3 channels
        if img is None:
            raise FileNotFoundError(f"Could not read image: {self.paths[i]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.tf(image=img)["image"], self.labels[i]


def build_model(pretrained=True):
    return timm.create_model("efficientnet_b0", pretrained=pretrained,
                             num_classes=2, drop_rate=0.3)


@torch.no_grad()
def predict_probs(model, loader, device):
    """Return (P(defective), true labels) for every image in the loader."""
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy()
        probs.append(p)
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)
