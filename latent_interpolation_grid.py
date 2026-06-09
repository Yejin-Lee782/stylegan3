import os
import click
import numpy as np
import torch
from PIL import Image

import dnnlib
import legacy


def save_image_grid(imgs, fname, grid_size):
    gw, gh = grid_size
    imgs = np.asarray(imgs)
    imgs = (imgs * 127.5 + 128).clip(0, 255).astype(np.uint8)
    imgs = imgs.transpose(0, 2, 3, 1)

    h, w = imgs.shape[1], imgs.shape[2]
    grid = Image.new("RGB", (gw * w, gh * h))

    for idx, img in enumerate(imgs):
        x = (idx % gw) * w
        y = (idx // gw) * h
        grid.paste(Image.fromarray(img, "RGB"), (x, y))

    grid.save(fname)


@click.command()
@click.option("--network", required=True)
@click.option("--outdir", required=True)
@click.option("--seed-a", type=int, default=10)
@click.option("--seed-b", type=int, default=100)
@click.option("--trunc", type=float, default=0.95)
@click.option("--steps", type=int, default=9)
def main(network, outdir, seed_a, seed_b, trunc, steps):
    os.makedirs(outdir, exist_ok=True)
    device = torch.device("cuda")

    print(f'Loading network from "{network}"...')
    with dnnlib.util.open_url(network) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)

    c = torch.empty([1, G.c_dim], device=device)

    rng_a = np.random.RandomState(seed_a)
    rng_b = np.random.RandomState(seed_b)

    z_a = torch.from_numpy(rng_a.randn(1, G.z_dim)).to(device)
    z_b = torch.from_numpy(rng_b.randn(1, G.z_dim)).to(device)

    with torch.no_grad():
        w_a = G.mapping(z_a, c, truncation_psi=trunc)
        w_b = G.mapping(z_b, c, truncation_psi=trunc)

    images = []
    alphas = torch.linspace(0, 1, steps, device=device)

    with torch.no_grad():
        for alpha in alphas:
            w = (1 - alpha) * w_a + alpha * w_b
            img = G.synthesis(w, noise_mode="const")
            images.append(img.cpu().numpy()[0])

    out_path = os.path.join(
        outdir,
        f"latent_interpolation_seed{seed_a}_to_seed{seed_b}_trunc{trunc}.png"
    )

    save_image_grid(images, out_path, grid_size=(steps, 1))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()