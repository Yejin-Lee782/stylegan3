from pathlib import Path
from PIL import Image
import cv2
import argparse
import numpy as np

VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

def center_crop_resize(frame, resolution):
    # OpenCV reads BGR, convert to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h, w = frame.shape[:2]
    side = min(h, w)

    y0 = (h - side) // 2
    x0 = (w - side) // 2

    crop = frame[y0:y0 + side, x0:x0 + side]
    img = Image.fromarray(crop)
    img = img.resize((resolution, resolution), Image.LANCZOS)
    return img

def get_frame_count(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count

def extract_frame(video_path, frame_idx, resolution):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return None

    return center_crop_resize(frame, resolution)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Video dataset folder")
    parser.add_argument("--dest", required=True, help="Output image folder")
    parser.add_argument("--target", type=int, default=32550)
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    videos = sorted([p for p in source.rglob("*") if p.suffix.lower() in VIDEO_EXTS])
    print("Number of videos:", len(videos))

    if len(videos) == 0:
        raise RuntimeError("No video files found.")

    saved = 0

    # If there are more videos than target images, use one middle frame per video.
    # If there are fewer videos, sample multiple frames per video.
    frames_per_video = max(1, args.target // len(videos) + 1)

    for video_path in videos:
        if saved >= args.target:
            break

        total_frames = get_frame_count(video_path)
        if total_frames <= 0:
            print("Skip broken video:", video_path)
            continue

        # sample frames from 20% to 80% range to avoid black intro/outro frames
        sample_count = min(frames_per_video, args.target - saved)
        indices = np.linspace(
            int(total_frames * 0.2),
            int(total_frames * 0.8),
            sample_count,
            dtype=int
        )

        for idx in indices:
            if saved >= args.target:
                break

            img = extract_frame(video_path, int(idx), args.resolution)
            if img is None:
                continue

            out_path = dest / f"{saved:06d}.png"
            img.save(out_path)
            saved += 1

            if saved % 1000 == 0:
                print("Saved:", saved)

    print("Final saved images:", saved)

    if saved != args.target:
        print("WARNING: target count not reached.")
        print("Expected:", args.target)
        print("Saved:", saved)

if __name__ == "__main__":
    main()
