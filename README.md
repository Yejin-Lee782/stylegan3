## Alias-Free Generative Adversarial Networks (StyleGAN3)

## Env Setup (Docker)

The Docker image is built from:

FROM nvcr.io/nvidia/pytorch:25.01-py3

Additional Python packages installed inside the container:

imageio==2.37.3
imageio-ffmpeg==0.4.4
pyspng==0.1.4

### 1. Build Docker Image

From the project root directory, run:

```
docker build -t stylegan3 .
```

### 2. Run Docker Container

To run the container with GPU support:
```
docker run --gpus all -it --rm \
    --ipc=host \
    -v $(pwd):/workspace \
    --workdir /workspace \
    stylegan3
```
And you want to start an interative bash shell inside the container:

```
sudo docker run --gpus all -it --rm \
     --ipc=host \
     --user $(id -u):$(id -g) \
     -v $(pwd):/workspace \
     -v $(pwd)/datasets:/workspace/datasets \
     --workdir /workspace \
     -e HOME=/workspace \
     stylegan3-metrics \
     bash
```
Note: Do not put a space after the backslash \.

## Dataset Tool

```
sudo docker run --gpus all -it --rm \
  --ipc=host \
  --user $(id -u):$(id -g) \
  -v $(pwd):/workspace \
  -v $(pwd)/datasets:/workspace/datasets \
  --workdir /workspace \
  -e HOME=/workspace \
  stylegan3 \
  python dataset_tool.py \
    --source=/datasets/YOUR_PATH \
    --dest=/datasets/YOUR_PATH.zip \
    --resolution=256x256
```

## Train 
Example command:
```
sudo docker run --gpus all -it --rm \
  --ipc=host \
  --user $(id -u):$(id -g) \
  -v $(pwd):/workspace \
  -v $(pwd)/datasets:/workspace/datasets \
  --workdir /workspace \
  -e HOME=/workspace \
  stylegan3 \
  python train.py \
    --outdir=/workspace/training-runs \
    --cfg=stylegan3-r \
    --data=/datasets/YOUR_PATH.zip \
    --gpus=1 \
    --batch=32 \
    --batch-gpu=8
    --gamma=2 \
    --mirror=1 \
    --kimg=5000 \
    --snap=10 \
    --metrics=none \
    --cbase=16384 \
    --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-256x256.pkl
```

## Generate Images

Example command:

```
sudo docker run --gpus all -it --rm \
  --ipc=host \
  --user $(id -u):$(id -g) \
  -v $(pwd):/workspace \
  --workdir /workspace \
  -e HOME=/workspace \
  stylegan3 \
  python gen_images.py \
    ---outdir=/workspace/generated_images \
    ---trunc=0.9
    --seeds=0-999
    --network=/workspace/training-runs/YOUR_MODEL_DIR/network-snapshot-XXXXXX.pkl
```

## Dataset and Checkpoints
datasets : https://drive.google.com/drive/folders/1ghEHozcdQRrskrWuQNxPsqHVfVjJR6Hp?usp=drive_link

Checkpoints :https://drive.google.com/drive/folders/1ghEHozcdQRrskrWuQNxPsqHVfVjJR6Hp?usp=drive_link

best checkpoint : https://drive.google.com/file/d/1i17xPAwcwWZb61CzrS5q-NAALjLI2Kuw/view?usp=drive_link

