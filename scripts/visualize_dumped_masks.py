#!/usr/bin/env python3
import os
import sys
import numpy as np
import cv2
from glob import glob
import imageio
import subprocess as sp
# RESNET: import these for slim version of resnet

def save_masked_image (path, image_path, masks_path):
    images = []

    image = cv2.imread(image_path, -1)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    assert len(image.shape) == 3
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(rgb)
    #vis = np.zeros_like(rgb, dtype=np.uint8)
    vis = np.copy(rgb).astype(np.float32)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    print(np.unique(mask))
    vis[:, :, 0][mask > 0.5] *= 0.5
    vis[:, :, 1][mask > 0.5] *= 0.5
    vis[:, :, 2][mask > 0.5] *= 0.5
    b, g, r = 180, 119, 31
    vis[:, :, 0] += b * mask * 0.5
    vis[:, :, 1] += g * mask * 0.5
    vis[:, :, 2] += r * mask * 0.5
    images.append(np.clip(vis, 0, 255).astype(np.uint8))
    imageio.mimsave(path + ".gif",images, duration = 1)
    sp.check_call('gifsicle --colors 256 -O3 < %s.gif > %s; rm %s.gif' % (path, path, path), shell=True)
    pass


for image_path in glob('picpac_dump/*_0_*.png'):
    mask_path = image_path.replace("_0_", "_1_")
    print(image_path, mask_path)
    if os.path.exists(mask_path):
        save_masked_image(image_path + '.gif', image_path, mask_path)



