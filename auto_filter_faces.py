import random
import shutil
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# Settings
# =========================
SOURCE_DIR = Path("/home/nuc/datasets/CelebV-HQ-images-256")
OUTPUT_DIR = Path("/home/nuc/dataset/filtered_dataset")

CLEAN_DIR = OUTPUT_DIR / "clean"
NOISY_DIR = OUTPUT_DIR / "noisy"
RANDOM_3000_DIR = OUTPUT_DIR / "random_3000_clean"

NUM_RANDOM_CLEAN = 3000
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Quality thresholds
MIN_FACE_CONFIDENCE = 0.60

# Face should occupy at least this ratio of image area
# Increase this if faces are too small.
MIN_FACE_AREA_RATIO = 0.04

# Blur threshold.
# Lower value = more permissive.
# For movie frames, 60~100 is reasonable.
BLUR_THRESHOLD = 70.0

# Brightness range.
# 0 = black, 255 = white
BRIGHTNESS_MIN = 40
BRIGHTNESS_MAX = 220

# Contrast threshold.
# Very low contrast images are often bad.
CONTRAST_THRESHOLD = 20


# =========================
# Prepare folders
# =========================
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
NOISY_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_3000_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Utility functions
# =========================
def variance_of_laplacian(gray):
    """
    Higher value means sharper image.
    Lower value means blurry image.
    """
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def classify_image(image_path, face_detector):
    img = cv2.imread(str(image_path))

    if img is None:
        return "noisy", "cannot_read", {}

    h, w = img.shape[:2]

    if h < 128 or w < 128:
        return "noisy", "too_small_image", {"width": w, "height": h}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = gray.mean()
    contrast = gray.std()
    blur_score = variance_of_laplacian(gray)

    if brightness < BRIGHTNESS_MIN:
        return "noisy", "too_dark", {
            "brightness": brightness,
            "contrast": contrast,
            "blur": blur_score,
        }

    if brightness > BRIGHTNESS_MAX:
        return "noisy", "too_bright", {
            "brightness": brightness,
            "contrast": contrast,
            "blur": blur_score,
        }

    if contrast < CONTRAST_THRESHOLD:
        return "noisy", "low_contrast", {
            "brightness": brightness,
            "contrast": contrast,
            "blur": blur_score,
        }

    if blur_score < BLUR_THRESHOLD:
        return "noisy", "blurry", {
            "brightness": brightness,
            "contrast": contrast,
            "blur": blur_score,
        }

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)

    if not results.detections:
        return "noisy", "no_face", {
            "brightness": brightness,
            "contrast": contrast,
            "blur": blur_score,
        }

    detections = results.detections

    if len(detections) > 1:
        return "noisy", "multiple_faces", {
            "num_faces": len(detections),
            "brightness": brightness,
            "contrast": contrast,
            "blur": blur_score,
        }

    det = detections[0]
    confidence = det.score[0]

    if confidence < MIN_FACE_CONFIDENCE:
        return "noisy", "low_face_confidence", {
            "face_confidence": confidence,
            "brightness": brightness,
            "contrast": contrast,
            "blur": blur_score,
        }

    bbox = det.location_data.relative_bounding_box

    x_min = bbox.xmin
    y_min = bbox.ymin
    box_w = bbox.width
    box_h = bbox.height

    face_area_ratio = box_w * box_h

    if face_area_ratio < MIN_FACE_AREA_RATIO:
        return "noisy", "face_too_small", {
            "face_area_ratio": face_area_ratio,
            "face_confidence": confidence,
            "brightness": brightness,
            "contrast": contrast,
            "blur": blur_score,
        }

    # Optional: reject strongly off-center faces
    face_center_x = x_min + box_w / 2
    face_center_y = y_min + box_h / 2

    if face_center_x < 0.20 or face_center_x > 0.80:
        return "noisy", "face_off_center_x", {
            "face_center_x": face_center_x,
            "face_area_ratio": face_area_ratio,
            "face_confidence": confidence,
            "brightness": brightness,
            "contrast": contrast,
            "blur": blur_score,
        }

    if face_center_y < 0.15 or face_center_y > 0.85:
        return "noisy", "face_off_center_y", {
            "face_center_y": face_center_y,
            "face_area_ratio": face_area_ratio,
            "face_confidence": confidence,
            "brightness": brightness,
            "contrast": contrast,
            "blur": blur_score,
        }

    return "clean", "passed", {
        "face_area_ratio": face_area_ratio,
        "face_confidence": confidence,
        "brightness": brightness,
        "contrast": contrast,
        "blur": blur_score,
    }


# =========================
# Main filtering
# =========================
image_files = [
    p for p in SOURCE_DIR.rglob("*")
    if p.suffix.lower() in IMAGE_EXTENSIONS
]

print(f"Total images found: {len(image_files)}")

records = []

mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=MIN_FACE_CONFIDENCE
) as face_detector:

    for idx, img_path in enumerate(tqdm(image_files)):
        label, reason, metrics = classify_image(img_path, face_detector)

        new_name = f"{idx:06d}{img_path.suffix.lower()}"

        if label == "clean":
            dst = CLEAN_DIR / new_name
        else:
            dst = NOISY_DIR / new_name

        shutil.copy2(img_path, dst)

        record = {
            "original_path": str(img_path),
            "saved_path": str(dst),
            "label": label,
            "reason": reason,
        }
        record.update(metrics)
        records.append(record)


# =========================
# Save report
# =========================
df = pd.DataFrame(records)
report_path = OUTPUT_DIR / "report.csv"
df.to_csv(report_path, index=False)

print("Filtering finished.")
print(f"Clean images: {(df['label'] == 'clean').sum()}")
print(f"Noisy images: {(df['label'] == 'noisy').sum()}")
print(f"Report saved to: {report_path}")


# =========================
# Randomly select 3000 clean images
# =========================
clean_images = [
    p for p in CLEAN_DIR.iterdir()
    if p.suffix.lower() in IMAGE_EXTENSIONS
]

print(f"Total clean images: {len(clean_images)}")

if len(clean_images) < NUM_RANDOM_CLEAN:
    print(
        f"Warning: only {len(clean_images)} clean images found. "
        f"Cannot select {NUM_RANDOM_CLEAN} images."
    )
else:
    random.seed(SEED)
    selected = random.sample(clean_images, NUM_RANDOM_CLEAN)

    for i, img_path in enumerate(selected):
        dst = RANDOM_3000_DIR / f"{i:05d}{img_path.suffix.lower()}"
        shutil.copy2(img_path, dst)

    print(f"Random {NUM_RANDOM_CLEAN} clean images saved to: {RANDOM_3000_DIR}")