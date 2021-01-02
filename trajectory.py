import numpy as np, math, sys, time, matplotlib.pyplot as plt
from phi_star import main as phi_star_gen
from utils import dist, over_sampling
from config import *
import plot

M_6 = (1 / np.math.factorial(5)) * np.array([[1,26,66,26,1,0], [-5,-50,0,50,5,0], [10,20,-60,20,10,0], [-10,20,0,-20,10,0], [5,-20,30,-20,5,0], [-1,5,-10,10,-5,1]])

# NUmber of points sampled from a single segment in a B-spline
UPSAMPLE_SCALE = 5

def bsplineUpsample(path):
    smoothed = np.zeros((len(path) * UPSAMPLE_SCALE, 2))

    for i in range(len(path)):
        p = np.array([path[np.clip(i + j - 2, 0, len(path)-1)] for j in range(6)]).reshape(6, 2)

        for uidx, u in enumerate(np.linspace(0, 1, UPSAMPLE_SCALE)):
            m = np.array([1, u, u*u, u*u*u, u*u*u*u, u*u*u*u*u]).reshape(1,6).dot(M_6)
            smoothed[i * UPSAMPLE_SCALE + uidx, :] = m.dot(p).reshape(-1)

    return smoothed

def main():
    path, grid_obs, start, goal = phi_star_gen()
    C = 3

    path = np.array(over_sampling([p.pos for p in path], max_length=1))
    path[5, :] = np.array([2, 5])
    path[13, :] = np.array([5, 17])
    path[20, :] = np.array([15, 2])

    smoothed = bsplineUpsample(path)

    smoothed = np.array(smoothed)

    plot.display(start, goal, grid_obs, path, smoothed, hold=True)
    

if __name__ == '__main__':
    main()