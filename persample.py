"""Per-sample metrics copied VERBATIM from Kinshuk's cldm/metrics.py (SSIM / LPIPS-VGG / ORB),
for consistency with his paper. Only the unused matplotlib + cldm.config imports are dropped
(they trigger side effects); the metric functions and module setup are unchanged.

All three take [B, C, H, W] tensors in [0, 1] and return np.ndarray[B]."""
from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch
import lpips
import cv2
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lpips_model = lpips.LPIPS(net='vgg').to(device).eval()
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def calculate_ssim_batch(original_images: torch.Tensor,
                         reconstructed_images: torch.Tensor) -> np.ndarray:
    """original_images, reconstructed_images: [B, C, H, W] in [0,1] -> np.ndarray [B] SSIM."""
    B, C, H, W = original_images.shape
    scores = []
    for i in range(B):
        orig = original_images[i].cpu().numpy()
        recon = reconstructed_images[i].cpu().numpy()
        orig = np.clip(orig, 0, 1)
        recon = np.clip(recon, 0, 1)
        orig = np.transpose(orig, (1, 2, 0))
        recon = np.transpose(recon, (1, 2, 0))
        if orig.shape[2] == 3:
            orig_gray = np.dot(orig[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            orig_gray = orig.squeeze(axis=2)
        if recon.shape[2] == 3:
            recon_gray = np.dot(recon[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            recon_gray = recon.squeeze(axis=2)
        score = ssim(orig_gray, recon_gray, data_range=1.0)
        scores.append(score)
    return np.array(scores)


def calculate_lpips_score_batch(imgs1: torch.Tensor, imgs2: torch.Tensor) -> np.ndarray:
    """imgs1, imgs2: [B, C, H, W] in [0,1] -> np.ndarray [B] LPIPS (VGG)."""
    if imgs1 is None or imgs2 is None:
        return np.full((0,), np.nan)
    scores = []
    for img1, img2 in zip(imgs1, imgs2):
        x1 = img1.unsqueeze(0).to(device).float()
        x2 = img2.unsqueeze(0).to(device).float()
        if x1.size(1) == 1:
            x1 = x1.repeat(1, 3, 1, 1)
            x2 = x2.repeat(1, 3, 1, 1)
        x1 = F.interpolate(x1, size=(256, 256), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=(256, 256), mode='bilinear', align_corners=False)
        if x1.min() >= 0 and x1.max() <= 1:
            x1 = x1 * 2 - 1
            x2 = x2 * 2 - 1
        with torch.no_grad():
            score = lpips_model(x1, x2).item()
        scores.append(score)
    return np.array(scores)


def calculate_orb_similarity_batch(imgs1: torch.Tensor, imgs2: torch.Tensor) -> np.ndarray:
    """imgs1, imgs2: [B, C, H, W] in [0,1] -> np.ndarray [B] ORB match-fraction."""
    if imgs1 is None or imgs2 is None:
        return np.full((0,), np.nan)
    scores = []
    for img1, img2 in zip(imgs1, imgs2):
        arr1 = (img1.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        arr2 = (img2.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        gray1 = cv2.cvtColor(arr1, cv2.COLOR_RGB2GRAY) if arr1.ndim == 3 else arr1
        gray2 = cv2.cvtColor(arr2, cv2.COLOR_RGB2GRAY) if arr2.ndim == 3 else arr2
        if gray2.shape != gray1.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        if des1 is None or des2 is None or not kp1 or not kp2:
            scores.append(0.0)
        else:
            matches = bf.match(des1, des2)
            scores.append(len(matches) / max(len(kp1), len(kp2)))
    return np.array(scores)
