import numpy as np, math, sys, time, matplotlib.pyplot as plt
from phi_star import main as phi_star_gen
from utils import dist, Node, lineOfSight, phi, lineOfSightNeighbors, corners, pathLength, updateGridBlockedCells, NoPathFound
from config import *

def main():
    res = phi_star_gen()
    print(res)

if __name__ == '__main__':
    main()