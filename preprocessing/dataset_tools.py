# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Patched for REPA-diffusion:
#   - encode: configurable --batch-size (default 8, not 1)
#   - encode: --device flag for per-GPU selection
#   - encode: --rank / --world-size for data-parallel sharding
#   - encode: writes shard-scoped dataset.json so merge_vae_shards.py can combine them
#   - removed WORLD_SIZE != 1 guard so the script can be launched from a shell loop

"""Tool for creating ZIP/PNG based datasets."""

from collections.abc import Iterator
from dataclasses import dataclass
import functools
import io
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import click
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

from encoders import StabilityVAEEncoder

#----------------------------------------------------------------------------

@dataclass
class ImageEntry:
    img: np.ndarray
    label: Optional[int]

#----------------------------------------------------------------------------

def parse_tuple(s: str) -> Tuple[int, int]:
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise click.ClickException(f'cannot parse tuple {s}')

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION

#----------------------------------------------------------------------------

def open_image_folder(source_dir, *, max_images: Optional[int]) -> tuple[int, Iterator[ImageEntry]]:
    input_images = []
    def _recurse_dirs(root: str):
        with os.scandir(root) as it:
            for e in it:
                if e.is_file():
                    input_images.append(os.path.join(root, e.name))
                elif e.is_dir():
                    _recurse_dirs(os.path.join(root, e.name))
    _recurse_dirs(source_dir)
    input_images = sorted([f for f in input_images if is_image_ext(f)])

    arch_fnames = {fname: os.path.relpath(fname, source_dir).replace('\\', '/') for fname in input_images}
    max_idx = maybe_min(len(input_images), max_images)

    labels = dict()
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            data = json.load(file)['labels']
            if data is not None:
                labels = {x[0]: x[1] for x in data}

    if len(labels) == 0:
        toplevel_names = {arch_fname: arch_fname.split('/')[0] if '/' in arch_fname else '' for arch_fname in arch_fnames.values()}
        toplevel_indices = {toplevel_name: idx for idx, toplevel_name in enumerate(sorted(set(toplevel_names.values())))}
        if len(toplevel_indices) > 1:
            labels = {arch_fname: toplevel_indices[toplevel_name] for arch_fname, toplevel_name in toplevel_names.items()}

    def iterate_images():
        for idx, fname in enumerate(input_images):
            img = np.array(PIL.Image.open(fname).convert('RGB'))
            yield ImageEntry(img=img, label=labels.get(arch_fnames[fname]))
            if idx >= max_idx - 1:
                break
    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_image_zip(source, *, max_images: Optional[int]) -> tuple[int, Iterator[ImageEntry]]:
    with zipfile.ZipFile(source, mode='r') as z:
        input_images = [str(f) for f in sorted(z.namelist()) if is_image_ext(f)]
        max_idx = maybe_min(len(input_images), max_images)
        labels = dict()
        if 'dataset.json' in z.namelist():
            with z.open('dataset.json', 'r') as file:
                data = json.load(file)['labels']
                if data is not None:
                    labels = {x[0]: x[1] for x in data}

    def iterate_images():
        with zipfile.ZipFile(source, mode='r') as z:
            for idx, fname in enumerate(input_images):
                with z.open(fname, 'r') as file:
                    img = np.array(PIL.Image.open(file).convert('RGB'))
                yield ImageEntry(img=img, label=labels.get(fname))
                if idx >= max_idx - 1:
                    break
    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def make_transform(transform, output_width, output_height):
    def scale(width, height, img):
        w, h = img.shape[1], img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img, 'RGB')
        ww = width if width is not None else w
        hh = height if height is not None else h
        return np.array(img.resize((ww, hh), PIL.Image.Resampling.LANCZOS))

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0]-crop)//2:(img.shape[0]+crop)//2,
                  (img.shape[1]-crop)//2:(img.shape[1]+crop)//2]
        img = PIL.Image.fromarray(img, 'RGB')
        return np.array(img.resize((width, height), PIL.Image.Resampling.LANCZOS))

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None
        img = img[(img.shape[0]-ch)//2:(img.shape[0]+ch)//2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = np.array(img.resize((width, height), PIL.Image.Resampling.LANCZOS))
        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width-height)//2:(width+height)//2, :] = img
        return canvas

    def center_crop_imagenet(image_size, arr):
        pil_image = PIL.Image.fromarray(arr)
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), PIL.Image.Resampling.BOX)
        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), PIL.Image.Resampling.BICUBIC)
        arr = np.array(pil_image)
        cy = (arr.shape[0] - image_size) // 2
        cx = (arr.shape[1] - image_size) // 2
        return arr[cy:cy+image_size, cx:cx+image_size]

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == 'center-crop':
        if output_width is None or output_height is None:
            raise click.ClickException('must specify --resolution=WxH when using ' + transform + ' transform')
        return functools.partial(center_crop, output_width, output_height)
    if transform == 'center-crop-wide':
        if output_width is None or output_height is None:
            raise click.ClickException('must specify --resolution=WxH when using ' + transform + ' transform')
        return functools.partial(center_crop_wide, output_width, output_height)
    if transform == 'center-crop-dhariwal':
        if output_width is None or output_height is None:
            raise click.ClickException('must specify --resolution=WxH when using ' + transform + ' transform')
        if output_width != output_height:
            raise click.ClickException('width and height must match')
        return functools.partial(center_crop_imagenet, output_width)
    assert False, 'unknown transform'

#----------------------------------------------------------------------------

def open_dataset(source, *, max_images: Optional[int]):
    if os.path.isdir(source):
        return open_image_folder(source, max_images=max_images)
    elif os.path.isfile(source):
        if file_ext(source) == 'zip':
            return open_image_zip(source, max_images=max_images)
        else:
            raise click.ClickException(f'Only zip archives are supported: {source}')
    else:
        raise click.ClickException(f'Missing input file or directory: {source}')

#----------------------------------------------------------------------------

def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)
    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname, data):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        os.makedirs(dest, exist_ok=True)
        def folder_write_bytes(fname, data):
            os.makedirs(os.path.dirname(os.path.join(dest, fname)), exist_ok=True)
            with open(os.path.join(dest, fname), 'wb') as f:
                f.write(data if isinstance(data, bytes) else data.encode())
        return '', folder_write_bytes, lambda: None

#----------------------------------------------------------------------------

@click.group()
def cmdline():
    pass

#----------------------------------------------------------------------------

@cmdline.command()
@click.option('--source',      help='Input directory or zip',    required=True,  metavar='PATH')
@click.option('--dest',        help='Output directory',          required=True,  metavar='PATH')
@click.option('--max-images',  help='Maximum number of images',  default=None,   type=int,    metavar='INT')
@click.option('--model-url',   help='VAE model URL or HF repo',  default='stabilityai/sd-vae-ft-mse', metavar='URL')
@click.option('--batch-size',  help='VAE batch size per forward pass (default 8)', default=8, type=int, metavar='INT')
@click.option('--device',      help='CUDA device, e.g. cuda:0 (default: cuda)',   default='cuda', metavar='STR')
@click.option('--rank',        help='Rank of this worker (0-indexed, default 0)',  default=0,    type=int, metavar='INT')
@click.option('--world-size',  help='Total number of parallel workers (default 1)',default=1,    type=int, metavar='INT')
def encode(
    model_url: str,
    source: str,
    dest: str,
    max_images: Optional[int],
    batch_size: int,
    device: str,
    rank: int,
    world_size: int,
):
    """Encode pixel data to VAE latents.

    Supports multi-GPU data-parallel encoding via --rank / --world-size.
    Each worker writes its shard to <dest>/shard_<rank>_of_<world_size>/.
    Run merge_vae_shards.py afterwards to combine shards into one directory.

    Example (4 GPUs):
        for i in 0 1 2 3; do
            python dataset_tools.py encode \
                --source /data/images --dest /data/vae-sd \
                --device cuda:$i --rank $i --world-size 4 &
        done
        wait
        python merge_vae_shards.py --dest /data/vae-sd --world-size 4
    """
    PIL.Image.init()
    if dest == '':
        raise click.ClickException('--dest must not be empty')
    if rank >= world_size:
        raise click.ClickException(f'--rank ({rank}) must be < --world-size ({world_size})')

    vae = StabilityVAEEncoder(vae_name=model_url, batch_size=batch_size)
    vae.init(torch.device(device))

    num_files, input_iter = open_dataset(source, max_images=max_images)

    # When sharding, each rank writes to its own sub-directory to avoid
    # filename collisions.  merge_vae_shards.py merges them afterwards.
    if world_size > 1:
        shard_dest = os.path.join(dest, f'shard_{rank:04d}_of_{world_size:04d}')
    else:
        shard_dest = dest

    archive_root_dir, save_bytes, close_dest = open_dest(shard_dest)

    labels = []
    desc = f'encode rank {rank}/{world_size} on {device}'

    for global_idx, image in tqdm(enumerate(input_iter), total=num_files, desc=desc):
        # Shard assignment: this rank handles global indices where
        # global_idx % world_size == rank
        if global_idx % world_size != rank:
            continue

        img_tensor = (
            torch.tensor(image.img)
            .to(device)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        mean_std = vae.encode_pixels(img_tensor)[0].cpu()

        idx_str = f'{global_idx:08d}'
        archive_fname = f'{idx_str[:5]}/img-mean-std-{idx_str}.npy'

        f = io.BytesIO()
        np.save(f, mean_std)
        save_bytes(os.path.join(archive_root_dir, archive_fname), f.getvalue())
        labels.append([archive_fname, image.label] if image.label is not None else None)

    metadata = {'labels': labels if all(x is not None for x in labels) else None}
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()
    print(f'[rank {rank}] Done. Wrote {len(labels)} latents to {shard_dest}')

#----------------------------------------------------------------------------

if __name__ == '__main__':
    cmdline()