#!/usr/bin/env python
from explorations import BaseSpaceExplorer
from explorers import OrientationSpaceExplorer
from copy import deepcopy
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def read_task(filepath, seq=0):
    """
    read source(start) and target(goal), and transform to right-hand and local coordinate system centered in source
    LCS: local coordinate system, or said vehicle-frame.
    GCS: global coordinate system
    """
    # read task and transform coordinate system to right-hand
    task = np.loadtxt('{}/{}_task.txt'.format(filepath, seq), delimiter=',')
    org, aim = task[0], task[1]
    source = BaseSpaceExplorer.CircleNode(x=org[0], y=-org[1], a=-np.radians(org[3]))  # coordinate of start in GCS
    target = BaseSpaceExplorer.CircleNode(x=aim[0], y=-aim[1], a=-np.radians(aim[3]))  # coordinate of goal in GCS
    # transform source and target coordinate from GCS to LCS.
    start = BaseSpaceExplorer.CircleNode(x=0, y=0, a=0)  # coordinate of start in LCS
    goal = deepcopy(target)
    goal.gcs2lcs(source)  # coordinate of goal in LCS
    return (source, target), (start, goal)


def read_grid(filepath, seq):
    # type: (str, int) -> np.ndarray
    """read occupancy grid map"""
    return np.array(Image.open('{}/{}_gridmap.png'.format(filepath, seq)))


def set_plot(explorer):
    # type: (OrientationSpaceExplorer) -> None
    plt.ion()
    plt.figure()
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().set_aspect('equal')
    plt.gca().set_facecolor((0.2, 0.2, 0.2))
    plt.gca().set_xlim((-30, 30))
    plt.gca().set_ylim((-30, 30))
    explorer.plot_grid(explorer.grid_map, explorer.grid_res)
    explorer.plot_circles([explorer.start, explorer.goal])
    plt.draw()


def main():
    filepath, seq = './test_scenes', 0
    (source, target), (start, goal) = read_task(filepath, seq)
    grid_map = read_grid(filepath, seq)
    grid_res = 0.1
    explorer = OrientationSpaceExplorer()
    explorer.initialize(start, goal, grid_map=grid_map, grid_res=grid_res)

    set_plot(explorer)
    print('Begin?')

    def plotter(circle):
        explorer.plot_circles([circle])
        plt.draw()
        raw_input('continue?')

    times = 1  # 100
    past = time.time()
    for i in range(times):
        if explorer.exploring(plotter=None):
            circle_path = explorer.circle_path
        else:
            print('Find No Path')
    now = time.time()
    explorer.plot_circles(circle_path)
    plt.draw()
    print('Runtime: {} ms (mean of {} times)'.format(np.round((now - past)/times, 4) * 1000, times))
    raw_input('Done')


if __name__ == '__main__':
    main()
