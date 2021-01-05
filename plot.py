import numpy as np, matplotlib.pyplot as plt, matplotlib.patches as patches, matplotlib.collections as collections
from utils import supercover, Node, lineOfSightNeighbors, lineOfSight, dist, phi
from config import DISPLAY_DELAY
from bspline import bsplineUpsample
from trajectory import OPTIMIZED_POINTS

fig, (ax, ax2) = plt.subplots(nrows=2)

getPos = lambda x: x.pos if isinstance(x, Node) else x

# originalPath: path coming from the global planner
# smoothPath: path after trajectory optimization
def display(start=None, goal=None, grid_obs=[], originalPath=[], optimPath=[], losses=[], currentOptimIdx=None, hold=False):
    ax.clear()
    ax.set_xlim(-0.5, grid_obs.shape[0])
    ax.set_ylim(-0.5, grid_obs.shape[0])
    
    ax.set_title('Trajectory')
    ax2.set_title('Loss function')

    obs = []
    x, y = np.mgrid[0:grid_obs.shape[0], 0:grid_obs.shape[1]]
    np.vectorize(lambda node, x, y: obs.append(patches.Rectangle([x, y], 1, 1)) if node == Node.OBSTACLE else None)(grid_obs, x, y)
    # obs = [patches.Rectangle([x, y], w, h) for x, y, w, h in extractRect(grid_obs)]
    ax.add_collection(collections.PatchCollection(obs))

    if start is not None:
        ax.add_patch(patches.Circle(getPos(start), .3, linewidth=1, facecolor='green'))
    if goal is not None:
        ax.add_patch(patches.Circle(getPos(goal), .3, linewidth=1, facecolor='blue'))
    
    if len(originalPath) > 0:
        smoothed = bsplineUpsample(originalPath)
        ax.plot(originalPath[:, 0], originalPath[:, 1], 'x', color='red', markersize=5)
        ax.plot(smoothed[:, 0], smoothed[:, 1], '-', color='red', linewidth=.8)
        if currentOptimIdx is not None:
            ax.plot(originalPath[currentOptimIdx:currentOptimIdx+OPTIMIZED_POINTS, 0], originalPath[currentOptimIdx:currentOptimIdx+OPTIMIZED_POINTS, 1], 'x', color='green', markersize=5)

    if len(optimPath) > 0:
        smoothed = bsplineUpsample(optimPath)
        ax.plot(optimPath[:, 0], optimPath[:, 1], 'o', color='magenta', markersize=5)
        ax.plot(smoothed[:, 0], smoothed[:, 1], '-', color='magenta', linewidth=.8)
        if currentOptimIdx is not None:
            ax.plot(optimPath[currentOptimIdx:currentOptimIdx+OPTIMIZED_POINTS, 0], optimPath[currentOptimIdx:currentOptimIdx+OPTIMIZED_POINTS, 1], 'o', color='green', markersize=5)

    if len(losses) > 0:
        ax2.clear()
        for loss in losses:
            ax2.plot(loss)
        # ax2.set_ylim([0, loss[0] + 10])

    if hold and isinstance(hold, bool):
        plt.show()
    else:
        plt.pause(DISPLAY_DELAY if isinstance(hold, bool) else hold)


def extractRect(grid):
    rects = []
    def alreadyDone(i, j):
        for x, y, w, h in rects:
            if x <= i < x+w and y <= j < y+h:
                return True
        return False

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not alreadyDone(i, j) and grid[i, j] == Node.OBSTACLE:
                for k in range(i+1, grid.shape[0]):
                    if grid[k, j] == Node.FREE:
                        break
                    k += 1
                imax = k
                for k in range(j+1, grid.shape[1]):
                    if grid[i:imax, j:k][grid[i:imax, j:k] == Node.FREE].size != 0:
                        break
                    k += 1
                jmax = k
                rects.append([i, j, imax-i, jmax-j])
    return rects




def waitForInput(obs, plotCb):
    refreshDisplay = False
    inputPending = True
    blockedCells = []

    def onclick(event):
        nonlocal refreshDisplay
        refreshDisplay = True
        x, y = int(event.xdata), int(event.ydata)
        obs[x, y] = Node.OBSTACLE if obs[x, y] == Node.FREE else Node.FREE
        blockedCells.append((x, y))

    def onkey(event):
        nonlocal refreshDisplay, inputPending
        if event.key == 'enter':
            refreshDisplay = True
            inputPending = False

    cid1 = fig.canvas.mpl_connect('button_press_event', onclick)
    cid2 = fig.canvas.mpl_connect('key_press_event', onkey)

    while inputPending:
        plt.title('Waiting for input... Press Enter to confirm')
        while not refreshDisplay:
            plt.pause(.001)
        refreshDisplay = False
        plotCb()
    
    fig.canvas.mpl_disconnect(cid1)
    fig.canvas.mpl_disconnect(cid2)
    return blockedCells
