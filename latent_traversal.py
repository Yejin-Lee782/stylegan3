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
@click.option("--network", required=True, help="Network pickle path")
@click.option("--outdir", required=True, help="Output directory")
@click.option("--seed", type=int, default=0, help="Base seed")
@click.option("--direction-seed", type=int, default=100, help="Random direction seed")
@click.option("--trunc", type=float, default=0.95, help="Truncation psi")
@click.option("--steps", type=int, default=7, help="Number of images")
@click.option("--strength", type=float, default=3.0, help="Traversal strength")
def main(network, outdir, seed, direction_seed, trunc, steps, strength):
    os.makedirs(outdir, exist_ok=True)

    device = torch.device("cuda")

    print(f'Loading network from "{network}"...')
    with dnnlib.util.open_url(network) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)

    # Base latent z
    rng = np.random.RandomState(seed)
    z = torch.from_numpy(rng.randn(1, G.z_dim)).to(device)

    # Mapping z -> w
    c = None
    if G.c_dim > 0:
        c = torch.zeros([1, G.c_dim], device=device)
    else:
        c = torch.empty([1, 0], device=device)

    with torch.no_grad():
        w = G.mapping(z, c, truncation_psi=trunc)

    # Random direction in W space
    rng_dir = np.random.RandomState(direction_seed)
    direction = torch.from_numpy(rng_dir.randn(*w.shape)).float().to(device)
    direction = direction / direction.norm()

    alphas = torch.linspace(-strength, strength, steps, device=device)

    images = []
    with torch.no_grad():
        for alpha in alphas:
            w_new = w + alpha * direction
            img = G.synthesis(w_new, noise_mode="const")
            images.append(img.cpu().numpy()[0])

    out_path = os.path.join(
        outdir,
        f"stylegan3-t_latent_traversal_seed{seed}_dir{direction_seed}_trunc{trunc}.png"
    )
    save_image_grid(images, out_path, grid_size=(steps, 1))

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()