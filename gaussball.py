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
    [0, 5]
])          
n_balls = 2


#### visualization parameters ####

view_w = 640
view_h = 480
color_comps = np.array([1., 0., 0.])


#### get principal components and use covariance scale to define pixel space transform ####

vals, vecs = linalg.eig(Cov)
pcmps = vecs * np.sqrt(vals)
pixels_per_unit = np.mean([view_w, view_h]) / (np.sqrt(vals.max()) * 4)

def pixel2plane(u, v) -> tuple:
    x = u - view_w / 2
    y = view_h / 2 - v
    x /= pixels_per_unit
    y /= pixels_per_unit
    return x, y

def plane2pixel(x, y) -> tuple:
    v = x * pixels_per_unit
    u = -y * pixels_per_unit
    v += view_w / 2
    u += view_h / 2
    return int(u), int(v)


#### get gaussian blur with principal components for background ####

idx = np.indices((view_w, view_h))
x, y = pixel2plane(idx[0], idx[1])
pts = np.dstack((x, y))
density = multivariate_normal.pdf(pts, cov=Cov)
greyscale = 1 - density / np.max(density)
# this is a hack to get an rgb image
background = np.empty((view_w, view_h, 3))
for c in range(3):
    background[:, :, c] = greyscale

origin = plane2pixel(0, 0)
cmp0 = plane2pixel(pcmps[0, 0], pcmps[0, 1])
cmp1 = plane2pixel(pcmps[1, 0], pcmps[1, 1])
cv2.line(background, origin, cmp0, color_comps)
cv2.line(background, origin, cmp1, color_comps)


cv2.imshow('image', background)#cv2.transpose(background))
cv2.waitKey()

