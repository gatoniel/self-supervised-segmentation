# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:04:46 2020

@author: niklas
"""

import os
import imageio
import tifffile
from pathlib import Path
import numpy as np
from bridson import poisson_disc_samples


def toy_circle_labels(
        width, height, radius,
        intensity_prob,
        noise_func
    ):
    labels = np.zeros((height, width), dtype=np.uint16)
    sketch = np.zeros_like(labels, dtype=float)
    
    # r is the distance between two center points, so we need to multiply by 2
    centers = poisson_disc_samples(width=width, height=height, r=radius * 2)
    
    ys = np.arange(height)
    xs = np.arange(width)
    
    meshy, meshx = np.meshgrid(ys, xs, indexing="ij")
    
    for i, (x, y) in enumerate(centers):
        dist = (meshx - x)**2 + (meshy - y)**2
        tmp_radius = np.random.uniform(radius / 2, radius)
        mask = dist < tmp_radius**2
        
        # enumerate starts at 0, but 0 is background
        labels[mask] = i + 1
        
        tmp_intensity = np.random.uniform(*intensity_prob)
        sketch[mask] = (
            tmp_radius - np.sqrt(dist[mask])
        ) / tmp_radius * tmp_intensity
        
    noise = noise_func(sketch)
    
    return labels, sketch, noise


def gaussian_noise(sigma):
    def func(sketch):
        return sketch + np.random.normal(0, sigma, size=sketch.shape)
    return func


path = r"D:\pytorch-CycleGAN-and-pix2pix\datasets"
newpath = os.path.join(path, "disk")

Path(newpath).mkdir(parents=True, exist_ok=True)

sets = ["val", "train", "test"]
num = 100

width = 600
height = 600
radius = 30

for s in sets:
    paths = []
    for b in ["A", "B", "C"]:
        pathb = os.path.join(newpath, s+b)
        Path(pathb).mkdir(parents=True, exist_ok=True)
        paths.append(pathb)
        
    for i in range(num):
        labels, sketch, noise = toy_circle_labels(
            width, height, radius,
            [1, 2],
            gaussian_noise(1.5 / 16)
        )
        for path, arr in zip(
                paths, [noise, sketch, labels]
        ):
            # tifffile.imsave(os.path.join(path, str(i)+".tif"), arr)
            imageio.imsave(os.path.join(path, str(i)+".jpg"), arr)