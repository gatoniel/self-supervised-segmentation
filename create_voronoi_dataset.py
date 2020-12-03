# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 13:45:03 2020

@author: niklas
"""

import os
import imageio
import scipy
import tifffile
from pathlib import Path
import numpy as np
from bridson import poisson_disc_samples
from scipy.spatial import Voronoi
import elasticdeform
from skimage.draw import polygon2mask
from scipy.ndimage.morphology import binary_opening
# from skimage.morphology import binary_opening
from skimage.morphology.selem import disk
from scipy.ndimage.morphology import distance_transform_edt

from multiprocessing import Pool
from functools import partial

def get_affinities(labels):
    shape = labels.shape
    # we use three channels, to save as rgb
    affinities = np.zeros((shape[0] - 1, shape[1] - 1, 3))
    
    top_affinity = (labels[:-1, :-1] - labels[1:, :-1]) == 0
    right_affinity = (labels[:-1, :-1] - labels[:-1, 1:]) == 0
    
    affinities[..., 0] = top_affinity
    affinities[..., 1] = right_affinity
    return affinities
    

def toy_voronoi_labels_affinities(
        width_, height_, radius, opening_radius, deform_sigma, deform_points,
        intensity_prob,
        noise_sigma
    ):
    # we will create bigger images such that we can crop out regions,
    # that look good
    
    offset = 2 * radius
    height = height_ + 2 * offset
    width = width_ + 2 * offset
    
    shape = (height, width)
    labels = np.zeros(shape, dtype=np.uint16)
    sketch = np.zeros_like(labels, dtype=float)
    
    # r is the distance between two center points, so we need to multiply by 2
    centers = poisson_disc_samples(width=width, height=height, r=radius * 2)
    
    # append far points to centers to complete voronoi regions
    x_max = width + 1
    y_max = height + 1
    centers = centers + [(-1, -1), (-1, y_max), (x_max, -1), (x_max, y_max)]
    
    vor = Voronoi(centers)
    
    # create selem with provided radius to apply clsoing
    selem = disk(opening_radius)
    for i, region in enumerate(vor.regions):
        if -1 in region or len(region) == 0:
            continue
        
        polygon = [vor.vertices[i] for i in region]
        mask = polygon2mask(shape, polygon)
        
        # close polygon mask with provided selem and radius
        mask = binary_opening(mask, selem)
        
        # enumerate starts at 0, but 0 is background
        labels[mask] = i + 1
        
        edt = distance_transform_edt(mask)
        edt = edt / edt.max()
        
        tmp_intensity = np.random.uniform(*intensity_prob)
        sketch[mask] = edt[mask] * tmp_intensity
        
    sketch = scipy.ndimage.gaussian_filter(sketch, radius / 4)
        
    [labels, sketch] = elasticdeform.deform_random_grid(
        [labels, sketch], sigma=deform_sigma, points=deform_points,
        # labels must be interpolated by nearest neighbor
        order=[0, 3],
        crop=(
            slice(offset, offset + height_ + 1),
            slice(offset, offset + width_ + 1)
        )
    )
    
    # labels = labels[offset:-offset + 1, offset:-offset + 1]
    # sketch = sketch[offset:-offset + 1, offset:-offset + 1]
    
    noise = sketch + np.random.normal(0, noise_sigma, size=sketch.shape)
    
    affinities = get_affinities(labels)
    
    return labels[:-1, :-1], sketch[:-1, :-1], noise[:-1, :-1], affinities



width = 600
height = 600
radius = 30
opening_radius = 10
deform_sigma = 10
deform_points = 8

# labels, sketch, noise, affinities = toy_voronoi_labels_affinities(
#     width, height, radius, opening_radius, deform_sigma, deform_points,
#     [1, 2],
#     gaussian_noise(1.5 / 16)
# )


if __name__ == '__main__':
    path = r"D:\pytorch-CycleGAN-and-pix2pix\datasets"
    newpath = os.path.join(path, "voronoi")
    
    Path(newpath).mkdir(parents=True, exist_ok=True)
    
    sets = ["val", "train", "test"]
    num = 100
    
    f = partial(
        toy_voronoi_labels_affinities,
        width, height, radius, opening_radius, deform_sigma, deform_points,
        [1, 2],
    )
    
    for s in sets:
        paths = []
        for b in ["A", "B", "C"]:
            pathb = os.path.join(newpath, s+b)
            Path(pathb).mkdir(parents=True, exist_ok=True)
            paths.append(pathb)
            
        with Pool(processes=18) as p:
            results = p.map(f, [1.5 / 16 for i in range(num)])
        
        for i in range(num):
            labels, sketch, noise, affinities = results[i]
            # we need to save noise as RGB, too, since cycleGAN needs
            # input and output channels to meet up
            noise = np.repeat(noise[..., np.newaxis], 3, axis=2)
            for path, arr in zip(
                    paths, [noise, affinities, labels]
            ):
                # tifffile.imsave(os.path.join(path, str(i)+".tif"), arr)
                imageio.imsave(os.path.join(path, str(i)+".jpg"), arr)