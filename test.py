#!/usr/bin/env python
from explorers import OrientationSpaceExplorer as OSExplorer
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
    source = OSExplorer.CircleNode(x=org[0], y=-org[1], a=-np.radians(org[3]))  # coordinate of start in GCS
    target = OSExplorer.CircleNode(x=aim[0], y=-aim[1], a=-np.radians(aim[3]))  # coordinate of goal in GCS
    a = source.x
    # transform source and target coordinate from GCS to LCS.
    start = OSExplorer.CircleNode(x=0., y=0., a=0.)  # coordinate of start in LCS
    goal = OSExplorer.CircleNode(x=aim[0], y=-aim[1], a=-np.radians(aim[3]))
    goal.gcs2lcs(source)  # coordinate of goal in LCS
    return (source, target), (start, goal)


def read_grid(filepath, seq):
    # type: (str, int) -> np.ndarray
    """read occupancy grid map"""
    return np.array(Image.open('{}/{}_gridmap.png'.format(filepath, seq)))


def set_plot(explorer):
    # type: (OSExplorer) -> None
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
    # preset
    filepath, seq = './test_scenes', 85
    (source, target), (start, goal) = read_task(filepath, seq)
    grid_map = read_grid(filepath, seq)
    grid_res = 0.1
    explorer = OSExplorer()
    explorer.initialize(start, goal, grid_map=grid_map, grid_res=grid_res)

    def plotter(circles):
        explorer.plot_circles(circles)
        plt.draw()
        raw_input('continue?')
    # set_plot(explorer)

    print('Begin?')
    map(explorer.exploring, [None])  # compile jit
    times = 10  # 100
    past = time.time()
    result = map(explorer.exploring, [None]*times)
    now = time.time()
    print('Runtime: {} ms (mean of {} times)'.format(np.round((now - past) / times, 4) * 1000, times))
    print('Done' if sum(result) else 'Find No Path')

    # explorer.plot_circles(explorer.circle_path)
    # plt.draw()
    # raw_input('Plotting')


if __name__ == '__main__':
    main()
