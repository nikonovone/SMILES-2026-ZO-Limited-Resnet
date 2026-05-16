"""
head_init.py — Initialize fc via L-BFGS logistic regression on 4-view TTA features.

Pipeline:
  1. Extract backbone features under 4 deterministic TTA views per image
     (base / zoom × identity / hflip).
  2. Treat each view as an independent training sample — 4N training pairs.
  3. L-BFGS multinomial logreg with fixed λ=75 (selected by prior sweep
     on 10% image-level holdout; see SOLUTION.md).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader

_DATA_DIR = "./data"
_NUM_CLASSES = 100
_FEATURE_BATCH_SIZE = 256
_LBFGS_MAX_ITER = 200
_LAMBDA = 75.0
_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _tta_transforms() -> list[T.Compose]:
    norm = T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD)
    return [
        T.Compose([T.Resize(224), T.ToTensor(), norm]),
        T.Compose([T.Resize(224), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), norm]),
        T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm]),
        T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), norm]),
    ]


@torch.no_grad()
def _extract_features_4view(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stacked features (4N × 512) and labels (4N,) across 4 TTA views."""
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.eval().to(device)

    feats_per_view: list[torch.Tensor] = []
    labels_single = torch.tensor(datasets.CIFAR100(root=_DATA_DIR, train=True, download=True).targets)

    for tf in _tta_transforms():
        ds = datasets.CIFAR100(root=_DATA_DIR, train=True, download=True, transform=tf)
        loader = DataLoader(ds, batch_size=_FEATURE_BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
        feats = []
        for x, _ in loader:
            feats.append(backbone(x.to(device, non_blocking=True)).cpu())
        feats_per_view.append(torch.cat(feats))

    del backbone
    feats_stacked = torch.cat(feats_per_view, dim=0)
    labels_stacked = labels_single.repeat(len(feats_per_view))
    return feats_stacked, labels_stacked


def _solve_logreg(
    phi: torch.Tensor, y: torch.Tensor, lam: float
) -> tuple[torch.Tensor, torch.Tensor]:
    n = phi.shape[0]
    phi = phi.float()
    W = torch.zeros(phi.shape[1], _NUM_CLASSES, requires_grad=True)
    b = torch.zeros(_NUM_CLASSES, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=_LBFGS_MAX_ITER,
                             history_size=20, tolerance_grad=1e-7,
                             tolerance_change=1e-9, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(phi @ W + b, y.long()) + 0.5 * lam * (W * W).sum() / n
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def init_last_layer(layer: nn.Linear) -> None:
    """Initialize fc via per-view 4-TTA features + L-BFGS logreg (λ=75)."""
    device = _select_device()
    print(f"[head_init] extracting 4-view TTA features on {device}")
    phi, y = _extract_features_4view(device)
    print(f"[head_init] solving L-BFGS logreg on {phi.shape[0]} samples, λ={_LAMBDA}")
    W, b = _solve_logreg(phi, y, _LAMBDA)
    with torch.no_grad():
        layer.weight.copy_(W.T.contiguous())
        layer.bias.copy_(b)
