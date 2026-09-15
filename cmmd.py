"""CMMD (CLIP Maximum Mean Discrepancy) — distribution metric, Jayasumana et al. CVPR 2024.
Stable at small n (no covariance to estimate, unlike FID). Uses the paper's exact CLIP model,
openai/clip-vit-large-patch14-336 (ViT-L/14 @ 336px) via transformers, so absolute values match
the original CMMD implementation Kinshuk used. Embeddings L2-normalized; Gaussian kernel
exp(-d^2/(2*sigma^2)) with sigma^2=10; unbiased MMD^2; x1000 scaling. LOWER = better.

The three constants below are the only things to change to reconcile with a different CMMD config."""
import numpy as np
import torch
from PIL import Image

_MODEL = "openai/clip-vit-large-patch14-336"   # paper's CLIP: ViT-L/14 @ 336px
_SIGMA2 = 10.0                                  # paper bandwidth (sigma^2)
_SCALE = 1000.0                                 # paper output scaling

try:
    from transformers import CLIPModel, CLIPImageProcessor
    _HAS = True
except Exception as _e:                         # pragma: no cover
    _HAS, _IMPORT_ERR = False, _e

_clip = {"model": None, "proc": None}


def available():
    return _HAS


def _get_clip(device):
    if _clip["model"] is None:
        _clip["model"] = CLIPModel.from_pretrained(_MODEL).to(device).eval()
        _clip["proc"] = CLIPImageProcessor.from_pretrained(_MODEL)   # resize->336 + CLIP normalize
    return _clip["model"], _clip["proc"]


@torch.no_grad()
def clip_embed_rgb(imgs_rgb, device, batch=32):
    """imgs_rgb: list of HxWx3 uint8 RGB arrays -> [N,768] L2-normalized CPU tensor."""
    model, proc = _get_clip(device)
    feats = []
    for i in range(0, len(imgs_rgb), batch):
        pil = [Image.fromarray(a) for a in imgs_rgb[i:i + batch]]
        px = proc(images=pil, return_tensors="pt")["pixel_values"].to(device)
        f = model.get_image_features(pixel_values=px).float()
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
    return torch.cat(feats, 0) if feats else torch.empty(0, 768)


def _mmd2(x, y):
    """Unbiased Gaussian-kernel MMD^2 between two [N,d]/[M,d] embedding tensors."""
    rx = (x * x).sum(1, keepdim=True)
    ry = (y * y).sum(1, keepdim=True)
    dxx = (rx + rx.t() - 2 * (x @ x.t())).clamp_min(0)
    dyy = (ry + ry.t() - 2 * (y @ y.t())).clamp_min(0)
    dxy = (rx + ry.t() - 2 * (x @ y.t())).clamp_min(0)
    g = lambda d: torch.exp(-d / (2 * _SIGMA2))
    n, m = x.shape[0], y.shape[0]
    kxx = (g(dxx).sum() - g(dxx).diag().sum()) / (n * (n - 1))
    kyy = (g(dyy).sum() - g(dyy).diag().sum()) / (m * (m - 1))
    kxy = g(dxy).mean()
    return kxx + kyy - 2 * kxy


def cmmd_from_embeds(er, eg):
    if er.shape[0] < 2 or eg.shape[0] < 2:
        return None
    return round(float(_SCALE * _mmd2(er, eg)), 4)


def cmmd_of(real_rgb, gen_rgb, device):
    """Two lists of preprocessed RGB uint8 arrays -> scalar CMMD (lower=better) or None if too few."""
    if not _HAS or len(real_rgb) < 2 or len(gen_rgb) < 2:
        return None
    return cmmd_from_embeds(clip_embed_rgb(real_rgb, device), clip_embed_rgb(gen_rgb, device))
