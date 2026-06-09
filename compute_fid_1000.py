import os
import argparse
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import linalg

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pytorch_fid.inception import InceptionV3


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def maybe_extract_zip(path: str, temp_dir: str) -> str:
    """
    If path is a zip file, extract it to a temporary directory.
    Otherwise, return the original path.
    """
    path = os.path.abspath(path)

    if os.path.isfile(path) and zipfile.is_zipfile(path):
        extract_dir = os.path.join(temp_dir, Path(path).stem)
        os.makedirs(extract_dir, exist_ok=True)

        print(f"[INFO] Extracting zip file: {path}")
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(extract_dir)

        return extract_dir

    return path


def list_images(path: str):
    """
    Recursively collect image files from a directory.
    """
    image_files = []

    for root, _, files in os.walk(path):
        for fname in files:
            ext = Path(fname).suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                image_files.append(os.path.join(root, fname))

    image_files = sorted(image_files)
    return image_files


class ImageDataset(Dataset):
    def __init__(self, image_files):
        self.image_files = image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = self.image_files[idx]

        img = Image.open(path).convert("RGB")
        img = np.asarray(img).astype(np.float32) / 255.0

        # HWC -> CHW
        img = torch.from_numpy(img.transpose(2, 0, 1))

        return img


def get_activations(image_files, model, batch_size, device, dims, num_workers):
    dataset = ImageDataset(image_files)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    pred_arr = np.empty((len(image_files), dims), dtype=np.float64)

    start_idx = 0

    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            pred = model(batch)[0]

            # If output is not 1x1, apply global average pooling
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()

            end_idx = start_idx + pred.shape[0]
            pred_arr[start_idx:end_idx] = pred
            start_idx = end_idx

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    FID formula:
    ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * sqrt(sigma1 sigma2))
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        print("[WARN] fid calculation produced singular product; adding eps.")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return float(fid)


def compute_fid(gen_path, real_path, batch_size, device, num_workers):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    with tempfile.TemporaryDirectory() as temp_dir:
        gen_path = maybe_extract_zip(gen_path, temp_dir)
        real_path = maybe_extract_zip(real_path, temp_dir)

        gen_images = list_images(gen_path)
        real_images = list_images(real_path)

        print(f"[INFO] Generated images: {len(gen_images)}")
        print(f"[INFO] Real images:      {len(real_images)}")

        if len(gen_images) == 0:
            raise RuntimeError("No generated images found.")

        if len(real_images) == 0:
            raise RuntimeError("No real images found.")

        if len(gen_images) != 1000:
            print(f"[WARN] Generated image count is {len(gen_images)}, not 1000.")

        print("[INFO] Extracting Inception features for generated images...")
        gen_act = get_activations(
            gen_images,
            model,
            batch_size=batch_size,
            device=device,
            dims=dims,
            num_workers=num_workers,
        )

        print("[INFO] Extracting Inception features for real images...")
        real_act = get_activations(
            real_images,
            model,
            batch_size=batch_size,
            device=device,
            dims=dims,
            num_workers=num_workers,
        )

        mu_gen = np.mean(gen_act, axis=0)
        sigma_gen = np.cov(gen_act, rowvar=False)

        mu_real = np.mean(real_act, axis=0)
        sigma_real = np.cov(real_act, rowvar=False)

        fid = calculate_frechet_distance(
            mu_gen,
            sigma_gen,
            mu_real,
            sigma_real,
        )

        return fid


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gen",
        type=str,
        required=True,
        help="Path to generated 1000 images folder or zip.",
    )

    parser.add_argument(
        "--real",
        type=str,
        required=True,
        help="Path to real reference image folder or zip.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    fid = compute_fid(
        gen_path=args.gen,
        real_path=args.real,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
    )

    print("========================================")
    print(f"FID-1000: {fid:.6f}")
    print("========================================")


if __name__ == "__main__":
    main()