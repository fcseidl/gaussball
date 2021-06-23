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
    [2, -1],
    [0, 5]
])          
n_balls = 2


#### specify colors ####

color_background = np.array([255, 255, 255])      # white
color_blur = np.array([255, 0, 0])
color_comps = np.array([0, 0, 255])


# simply plot density with principal components

asp = (800, 800)

# get principal components
vals, vecs = linalg.eig(Cov)
pcmps = vecs * np.sqrt(vals)
scale = np.sqrt(vals.max()) * 4 / np.mean(asp)

def pixel2plane(u, v):
    x = u - asp[0] / 2
    y = v - asp[1] / 2
    x *= scale
    y *= scale
    return x, y

def plane2pixel(x, y):
    u = x / scale
    v = y / scale
    u += asp[0] / 2
    v += asp[1] / 2
    return u, v

# background is gaussian blur
idx = np.indices(asp)
x, y = pixel2plane(idx[0], idx[1])
pts = np.dstack((x, y))
density = multivariate_normal.pdf(pts, cov=Cov)
#img = (1 - density / np.max(density))
greyscale = (1 - density / np.max(density))
img = np.empty((asp[0], asp[1], 3))
for c in range(3):
    img[:, :, c] = greyscale

# overlay principal components
origin = plane2pixel(0, 0)
cmp0 = plane2pixel(pcmps[0, 0], pcmps[0, 1])
cmp1 = plane2pixel(pcmps[1, 0], pcmps[1, 1])

cv2.imshow('image', img)
cv2.waitKey()
