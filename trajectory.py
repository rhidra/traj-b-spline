import math, sys, time, matplotlib.pyplot as plt, plot, nlopt, autograd.numpy as np, autograd
from phi_star import main as phi_star_gen
from utils import dist, over_sampling, Node
from bspline import M_6, bspline, bsplineAcc, bsplineVel, bsplineJerk, bsplineSnap, bsplineUpsample, extractPts
from config import *


# Number of points being optimized at the same time
OPTIMIZED_POINTS = 3

# Teta in the distance function in the paper
OBSTACLE_DISTANCE_THRESHOLD = 5

# Initial offset in the path when the optimization starts
INITIAL_OPTIM_OFFSET = 6

# Objective function parameters
lambda_p, lambda_v = .06, .02 # Endpoint cost position and velocity weights
lambda_c = 100 # Collision cost weight

# For the quadratic cost computation (E_q)
Q_2 = lambda dt: np.array([[0]*6,[0]*6, [0,0,4,6,8,10], [0,0,6,12,18,24], [0,0,8,18,28.8,40], [0,0,10,24,40,400/7]]) / (dt*dt*dt)
Q_3 = lambda dt: np.array([[0]*6,[0]*6,[0]*6, [0,0,0,36,72,120], [0,0,0,72,192,360], [0,0,0,120,360,720]]) / (dt*dt*dt*dt*dt)
Q_4 = lambda dt: np.array([[0]*6,[0]*6,[0]*6,[0]*6, [0,0,0,0,576,1440], [0,0,0,0,1440,4800]]) / (dt*dt*dt*dt*dt*dt*dt)


def norm(x):
    return np.sqrt(np.dot(x.T, x))


def euclideanDistanceTransform(grid_obs):
    edt = np.zeros(grid_obs.shape)
    x, y = np.meshgrid(np.arange(grid_obs.shape[0]), np.arange(grid_obs.shape[1]))
    for pt in zip(x.reshape(-1), y.reshape(-1)):
        i0, i1 = math.floor(pt[0] - 1 - OBSTACLE_DISTANCE_THRESHOLD), math.floor(pt[0] + 1 + OBSTACLE_DISTANCE_THRESHOLD)
        i0, i1 = np.clip(i0, 0, grid_obs.shape[0]-1), np.clip(i1, 0, grid_obs.shape[0]-1)
        j0, j1 = math.floor(pt[1] - 1 - OBSTACLE_DISTANCE_THRESHOLD), math.floor(pt[1] + 1 + OBSTACLE_DISTANCE_THRESHOLD)
        j0, j1 = np.clip(j0, 0, grid_obs.shape[1]-1), np.clip(j1, 0, grid_obs.shape[1]-1)
        score = np.inf

        for i, row in enumerate(grid_obs[i0:i1]):
            for j, val in enumerate(row[j0:j1]):
                if val == Node.OBSTACLE:
                    d = dist(pt, np.array([i+i0, j+j0]))
                    if d < score:
                        score = d
        edt[pt] = score
    return edt


# Cost function for a quintic B-spline starting at startIdx
# pts and globalPts should contain all the points from the optimized trajectory and global path
def cost(pts, globalPts, startIdx, distObs, delta_t):
    print('startIdx', startIdx)

    # Endpoint cost
    posDiff = bspline(0, extractPts(pts, startIdx + 5)) - globalPts[startIdx + 5]
    velDiff = bsplineVel(0, extractPts(pts, startIdx + 5), delta_t) - np.array([1, 1]) # Hardcoded expected velocity
    E_ep = lambda_p * np.dot(posDiff, posDiff) + lambda_v * np.dot(velDiff, velDiff)

    # Collision cost
    u = np.linspace(0, 1, 10)
    samples = np.vstack((np.repeat(np.arange(6), len(u)), np.tile(u, 6)))
    
    def computeDist(sample):
        p = bspline(sample[1], extractPts(pts, sample[0] + startIdx))
        return distObs[np.clip(np.int(p[0]), 0, distObs.shape[0]-1), np.clip(np.int(p[1]), 0, distObs.shape[1]-1)]
    distances = np.apply_along_axis(computeDist, 0, samples)
    mask = distances <= OBSTACLE_DISTANCE_THRESHOLD
    distances[mask] = np.square(distances[mask] - OBSTACLE_DISTANCE_THRESHOLD) / (2 * OBSTACLE_DISTANCE_THRESHOLD)
    distances[np.invert(mask)] = 0

    def computeVelocities(sample):
        p = bsplineVel(sample[1], extractPts(pts, sample[0] + startIdx), delta_t)
        return norm(p)
    velocities = np.apply_along_axis(computeVelocities, 0, samples)
    E_c = lambda_c * np.sum(np.dot(distances, velocities)) / (len(u) * 6)

    # Squared derivative cost
    lambda_q2, lambda_q3, lambda_q4 = .1e-3, .1e-3, .1e-3
    q2, q3, q4 = Q_2(delta_t), Q_3(delta_t), Q_4(delta_t)
    E_q = 0
    for i in range(6):
        A = np.dot(M_6, extractPts(pts, startIdx + i))
        B = A.T
        E_q = E_q + np.sum(lambda_q2 * np.dot(np.dot(B, q2), A) + lambda_q3 * np.dot(np.dot(B, q3), A) + lambda_q4 * np.dot(np.dot(B, q4), A))
    
    # Derivative limit cost
    max_vel, max_acc, max_jerk, max_snap = np.array([1000, 1000]), np.array([1000, 1000]), np.array([1000, 1000]), np.array([10000, 10000])
    u = np.linspace(0, 1, 10)
    samples = np.vstack((np.repeat(np.arange(6), len(u)), np.tile(u, 6)))
    def derivativeCost(pFunc, max_p, delta_t):
        def f(sample):
            p = pFunc(sample[1], extractPts(pts, sample[0] + startIdx), delta_t)
            norm_max = norm(max_p)
            norm_p = norm(p)
            return np.exp(norm_p - norm_max) - 1 if norm_p > norm_max else 0
        return f

    E_l = 0
    for sample in zip(samples[0], samples[1]):
        E_l = E_l + derivativeCost(bsplineVel, max_vel, delta_t)(sample)
        E_l = E_l + derivativeCost(bsplineAcc, max_acc, delta_t)(sample)
        E_l = E_l + derivativeCost(bsplineJerk, max_jerk, delta_t)(sample)
        E_l = E_l + derivativeCost(bsplineSnap, max_snap, delta_t)(sample)

    # Total cost
    E = E_ep + E_c + E_q + E_l

    if not isinstance(E_ep, autograd.numpy.numpy_boxes.ArrayBox):
        print('{} | {} | {} | {} => {}'.format(E_ep, E_c, E_q, E_l, E))
    return E


def optimTrajectory(path, distObs, grid_obs, trajDuration):
    path = np.pad(path, ((0, 6), (0, 0)), mode='edge')
    optim = np.copy(path)
    delta_t = trajDuration / len(path)
    # For each iteration, we optimize between [startOptim, startOptim + OPTIMIZED_POINTS]
    startOptim = INITIAL_OPTIM_OFFSET

    # NLopt configuration
    def objFun(optim, globalPath, startOptim, distObs, delta_t):
        def E(x):
            inp = np.vstack((optim[:startOptim], x.reshape(OPTIMIZED_POINTS, 2), optim[startOptim+OPTIMIZED_POINTS:]))
            return cost(inp, globalPath, startOptim - (6 - OPTIMIZED_POINTS), distObs, delta_t)
        gradE = autograd.grad(E)
        return E, gradE

    epochs = 20

    losses = np.zeros((len(path) - 6 - startOptim - OPTIMIZED_POINTS, epochs))
    while startOptim + OPTIMIZED_POINTS <= len(path) - 6:
        f, df = objFun(optim, path, startOptim, distObs, delta_t)
        x = optim[startOptim:startOptim+OPTIMIZED_POINTS].reshape(-1)

        beta = .9 # Momentum optimization param
        lr = 1.5e-1
        v = 0
        prev = np.zeros(6)
        prev[:] = x
        for i in range(epochs):
            print()
            losses[startOptim-INITIAL_OPTIM_OFFSET, i] = f(x)
            dx = df(x)
            v = beta * v + (1 - beta) * dx
            x = x - lr * v
            print('[{}] loss: {} | diff: {} | grad: {}'.format(i, losses[startOptim-INITIAL_OPTIM_OFFSET, i], np.linalg.norm(prev - x), dx))
            if i%3==0 or True:
                optim[startOptim:startOptim+OPTIMIZED_POINTS, :] = x.reshape(-1, 2)
                plot.display(None, None, grid_obs, path, optim, losses, delta_t=delta_t, currentOptimIdx=startOptim, hold=.1)
            if np.linalg.norm(prev - x) < .001 and False:
                print('Reached !')
                break
            prev[:] = x
        optim[startOptim:startOptim+OPTIMIZED_POINTS, :] = x.reshape(-1, 2)
        startOptim += 1
    return optim


def main():
    path, grid_obs, start, goal = phi_star_gen()

    path = np.array(over_sampling([p.pos for p in path], max_length=1))
    # path[5, :] = np.array([2, 5])
    # path[13, :] = np.array([5, 17])
    # path[20, :] = np.array([15, 2])

    # path = np.array([[start[0], start[1]], [goal[0], goal[1]]])
    # path = over_sampling(path, max_length=1)

    distObs = euclideanDistanceTransform(grid_obs)

    pathOptimized = optimTrajectory(path, distObs, grid_obs, trajDuration=10)

    print(path, pathOptimized)

    smoothed = bsplineUpsample(pathOptimized)

    
if __name__ == '__main__':
    main()