"""
This module attempts to solve the following type of optimization problem:
Given a (Gaussian) distribution and a fixed N, find N unit balls maximizing
the probability that a sample from the distribution will be in one of the 
balls. 
"""

import numpy as np
from numpy import linalg
from scipy.stats import multivariate_normal
from simanneal import Annealer

import cv2                      


#### specify problem instance ####

dim = 2                     # TODO: what about different dimensions?
Cov = np.array([
    [1, 0],
    [0, 5]
])          
n_balls = 2

# empirically estimate uncovered mass
def cost(soln, n_samp=100000) -> float:
    centers = soln.reshape(-1, dim)
    x = multivariate_normal.rvs(cov=Cov, size=n_samp)
    excluded = np.ones(n_samp)
    for c in centers:
        excluded *= (linalg.norm(c - x, axis=1) > 1)
    return excluded.sum() / n_samp


#### visualization parameters ####

view_w = 640
view_h = 480
color_comps = np.array([1., 0., 0.])    # blue
color_best = np.array([0., 1., 0.])     # green
color_current = np.array([0., 0., 1.])  # red


#### get principal components and use covariance scale to define pixel space transform ####
# NOTE: we work with transposed images because opencv's image coordinate scheme is moronic

vals, vecs = linalg.eig(Cov)
pcmps = vecs * np.sqrt(vals)
pixels_per_unit = int(np.mean([view_w, view_h]) / (np.sqrt(vals.max()) * 4))

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


#### wrap cost in functor which visualizes search ####

class Solver(Annealer):

    _show_every = 50

    def __init__(self, init_state) -> None:
        self._n_steps = 0
        self._min = np.infty
        self._argmin = None
        self.copy_strategy='method'         # numpy arrays have a deep copy method
        super(Solver, self).__init__(init_state)

    def move(self) -> None:
        self.state += 2e-2 * np.random.randn(*(self.state.shape)) - 5e-4 * self.state     # TODO: magic numbers

    def energy(self) -> float:
        c = cost(self.state)
        if c < self._min:
            #print('found solution with cost of', c)
            self._min = c
            self._argmin = self.state.copy()
        if self._n_steps % self._show_every == 0:
            img = background.copy()
            for center in self.state:
                u, v = plane2pixel(center[0], center[1])
                cv2.circle(img, (u,v), pixels_per_unit, color_current)
            for center in self._argmin:
                u, v = plane2pixel(center[0], center[1])
                cv2.circle(img, (u,v), pixels_per_unit, color_best)
            cv2.imshow('Optimizing...', cv2.transpose(img))
            cv2.waitKey(1)
        self._n_steps += 1
        return c


#### perform optimization ####

x0 = multivariate_normal.rvs(cov=Cov, size=n_balls)
x0 = np.atleast_2d(x0)
solver = Solver(init_state=x0)
centers, c = solver.anneal()
print('Best solution:\n', centers)
print('Cost:', c)

cv2.destroyAllWindows()
img = background.copy()
for center in centers:
    u, v = plane2pixel(center[0], center[1])
    cv2.circle(img, (u,v), pixels_per_unit, color_best)
cv2.imshow('Best solution found, with cost of %f' % c, cv2.transpose(img))
cv2.waitKey()
cv2.destroyAllWindows()