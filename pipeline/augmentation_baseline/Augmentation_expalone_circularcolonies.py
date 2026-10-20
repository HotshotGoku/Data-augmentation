"""Basic rotation augmentation for circular colony images.

This script creates 100 rotated copies per image (default) using OpenCV.
No Hough transform, cropping, or brightness/contrast adjustment is applied.
"""

import os
import cv2
import numpy as np

from utils.local_config import KRISTEN_EXP_FOLDER_2SP_FILTERED_RENAMED


def rotate_with_inscribed_circle_mask(img: np.ndarray, angle: float) -> np.ndarray:
	"""Rotate image around center and keep only the inscribed circular region."""
	h, w = img.shape[:2]
	cx, cy = w // 2, h // 2
	radius = min(cx, cy)

	rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
	rotated = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)

	mask = np.zeros((h, w), dtype=np.uint8)
	cv2.circle(mask, (cx, cy), radius, 255, -1)
	return cv2.bitwise_and(rotated, rotated, mask=mask)


def augment_folder_by_rotation(
	input_folder: str,
	output_folder: str,
	n_rotations: int = 100,
	valid_exts=(".jpg", ".tiff"),
) -> None:
	os.makedirs(output_folder, exist_ok=True)

	files = sorted(
		f for f in os.listdir(input_folder)
		if os.path.isfile(os.path.join(input_folder, f))
		and f.lower().endswith(valid_exts)
	)

	angles = np.linspace(0, 360, n_rotations, endpoint=False)

	for i, filename in enumerate(files, start=1):
		src_path = os.path.join(input_folder, filename)
		img = cv2.imread(src_path, cv2.IMREAD_COLOR)
		if img is None:
			print(f"[WARN] Skipping unreadable file: {filename}")
			continue

		stem, ext = os.path.splitext(filename)
		for angle in angles:
			rotated = rotate_with_inscribed_circle_mask(img, float(angle))
			out_name = f"{stem}_rot{angle:.1f}{ext}"
			out_path = os.path.join(output_folder, out_name)
			cv2.imwrite(out_path, rotated)

		if i % 25 == 0 or i == len(files):
			print(f"Processed {i}/{len(files)} images")

	print(f"Done. Augmented images written to: {output_folder}")


if __name__ == "__main__":
	input_folder = KRISTEN_EXP_FOLDER_2SP_FILTERED_RENAMED
	output_folder = f"{input_folder.rstrip('/')}_AUG100"
	augment_folder_by_rotation(input_folder=input_folder, output_folder=output_folder, n_rotations=100)

