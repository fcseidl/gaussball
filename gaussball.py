"""
This module attempts to solve the following type of optimization problem:
Given a (Gaussian) distribution and a fixed N, find N unit balls maximizing
the probability that a sample from the distribution will be in one of the 
balls. 
"""

import numpy as np
from numpy import linalg
from scipy.stats import multivariate_normal

import cv2                      


#### specify problem instance ####

dim = 2                     # TODO: what about different dimensions?
Cov = np.array([
    [1, 0],
    [0, 1]
])          
n_balls = 2


#### visualization parameters ####

asp = (800, 900)        # h, w
color_comps = np.array([1., 0., 0.])


#### get principal components and use covariance scale to define pixel space transform ####

vals, vecs = linalg.eig(Cov)
pcmps = vecs * np.sqrt(vals)
pixels_per_unit = np.mean(asp) / (np.sqrt(vals.max()) * 4)

def pixel2plane(u, v) -> tuple:
    x = u - asp[1] / 2
    y = asp[0] / 2 - v
    x /= pixels_per_unit
    y /= pixels_per_unit
    return x, y

def plane2pixel(x, y) -> tuple:
    u = x * pixels_per_unit
    v = -y * pixels_per_unit
    u += asp[1] / 2
    v += asp[0] / 2
    return int(u), int(v)


#### get gaussian blur with principal components for background ####

idx = np.indices(asp)
x, y = pixel2plane(idx[1], idx[0])
pts = np.dstack((x, y))
density = multivariate_normal.pdf(pts, cov=Cov)
greyscale = 1 - density / np.max(density)
# this is a hack to get an rgb image
background = np.empty((asp[0], asp[1], 3))
for c in range(3):
    background[:, :, c] = greyscale

origin = plane2pixel(0, 0)
cmp0 = plane2pixel(pcmps[0, 0], pcmps[0, 1])
cmp1 = plane2pixel(pcmps[1, 0], pcmps[1, 1])
cv2.line(background, origin, cmp0, color_comps)
cv2.line(background, origin, cmp1, color_comps)


cv2.imshow('image', background)
cv2.waitKey()

