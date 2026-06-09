import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SRC_DIR = Path("/home/nuc/datasets/CelebV-HQ-images-256")
DST_DIR = Path("/home/nuc/datasets/refined_faces")
REJECT_DIR = Path("/home/nuc/datasets/rejected_faces")

DST_DIR.mkdir(parents=True, exist_ok=True)
REJECT_DIR.mkdir(parents=True, exist_ok=True)

MIN_SIZE = 256
BLUR_THRESHOLD = 50.0
DARK_THRESHOLD = 25
BRIGHT_THRESHOLD = 220
DUPLICATE_HAMMING_THRESHOLD = 4


def dhash(image, hash_size=8):
    image = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(image)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return sum([2 ** i for i, v in enumerate(diff.flatten()) if v])


def hamming_distance(a, b):
    return bin(a ^ b).count("1")


def is_blurry(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < BLUR_THRESHOLD, score


def is_bad_brightness(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean = gray.mean()
    return mean < DARK_THRESHOLD or mean > BRIGHT_THRESHOLD, mean


hashes = []
kept = 0
rejected = 0

image_paths = []
for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
    image_paths.extend(SRC_DIR.rglob(ext))

for path in image_paths:
    try:
        pil_img = Image.open(path).convert("RGB")
    except Exception:
        shutil.copy2(path, REJECT_DIR / path.name)
        rejected += 1
        continue

    w, h = pil_img.size
    if min(w, h) < MIN_SIZE:
        shutil.copy2(path, REJECT_DIR / path.name)
        rejected += 1
        continue

    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    blurry, blur_score = is_blurry(img_bgr)
    if blurry:
        shutil.copy2(path, REJECT_DIR / path.name)
        rejected += 1
        continue

    bad_brightness, brightness = is_bad_brightness(img_bgr)
    if bad_brightness:
        shutil.copy2(path, REJECT_DIR / path.name)
        rejected += 1
        continue

    hval = dhash(pil_img)
    is_duplicate = any(hamming_distance(hval, old_h) <= DUPLICATE_HAMMING_THRESHOLD for old_h in hashes)

    if is_duplicate:
        shutil.copy2(path, REJECT_DIR / path.name)
        rejected += 1
        continue

    hashes.append(hval)

    out_name = f"{kept:06d}.png"
    pil_img.save(DST_DIR / out_name)
    kept += 1

print(f"Kept: {kept}")
print(f"Rejected: {rejected}")
print(f"Saved to: {DST_DIR}")