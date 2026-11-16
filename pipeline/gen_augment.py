"""Runtime generator-training augmentation (Kinshuk feedback, 2026-11-16).

augment_pair applies the SAME rotation + flip + brightness/contrast to BOTH the source (hint) and the
target, so the hint->target pair stays geometrically aligned. Used by reptorep_dataset.py when GEN_AUG=1.

Per-dataset config (set by the launcher via env, read in reptorep_dataset.py):
  - brightness=False for FLUORESCENT datasets (mux, kl): intensity/color IS the label -> freeze photometrics.
  - fill = rotation border colour: WHITE (255) for brightfield (initialstage_mod) + rattray,
           BLACK (0) for swarmevo (dish_remove).

Prototyped + visually signed off in datasets/preprocess/make_review_montages.py (aug_preview).
"""
import cv2
import numpy as np


def _rot(img, angle, fill):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


def _photo(img, bright, contrast):
    f = (img.astype(np.float32) - 128) * contrast + 128   # contrast about mid-grey
    f *= bright                                            # then brightness scale
    return np.clip(f, 0, 255).astype(np.uint8)


def augment_pair(src, tgt, rng, rotate=True, flip=True, brightness=True, fill=(255, 255, 255)):
    """src, tgt: HxWx3 uint8 RGB. rng: np.random.Generator. Returns (aug_src, aug_tgt, meta_str).
    The SAME sampled params are applied to src and tgt so the pair stays aligned."""
    ang = float(rng.uniform(0, 360)) if rotate else 0.0
    flp = bool(rng.random() < 0.5) if flip else False
    br = float(rng.uniform(0.85, 1.15)) if brightness else 1.0
    ct = float(rng.uniform(0.85, 1.15)) if brightness else 1.0
    out = []
    for im in (src, tgt):
        a = _rot(im, ang, fill) if rotate else im
        if flp:
            a = np.ascontiguousarray(a[:, ::-1])
        if brightness:
            a = _photo(a, br, ct)
        out.append(a)
    return out[0], out[1], f"rot={ang:.0f} flip={'T' if flp else 'F'} b={br:.2f} c={ct:.2f}"


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    g = (np.ones((256, 256, 3)) * np.arange(256).reshape(1, 256, 1)).astype(np.uint8)  # gradient
    a, b, meta = augment_pair(g, g.copy(), rng)
    assert a.shape == b.shape == (256, 256, 3)
    assert np.array_equal(a, b), "src and tgt must receive IDENTICAL augmentation (alignment)"
    a0, b0, _ = augment_pair(g, g.copy(), rng, rotate=False, flip=False, brightness=False)
    assert np.array_equal(a0, g), "no-op aug must return the source unchanged"
    print("gen_augment self-check OK:", meta)
