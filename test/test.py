#!/usr/bin/env python
from pySEA.explorers import OrientationSpaceExplorer as OSExplorer
from copy import deepcopy
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt


def center2rear(circle, wheelbase=2.96):  # type: (OSExplorer.CircleNode, float) -> OSExplorer.CircleNode
    """calculate the coordinate of rear track center according to mass center"""
    theta, r = circle.a + np.pi, wheelbase/2.
    circle.x += r * np.cos(theta)
    circle.y += r * np.sin(theta)
    return circle


def read_task(filepath, seq=0):
    """
    read source(start) and target(goal), and transform to right-hand and local coordinate system centered in source
    LCS: local coordinate system, or said vehicle-frame.
    GCS: global coordinate system
    """
    # read task and transform coordinate system to right-hand
    task = np.loadtxt('{}/{}_task.txt'.format(filepath, seq), delimiter=',')
    org, aim = task[0], task[1]
    # coordinate of the center of mass on source(start) state, in GCS
    source = OSExplorer.CircleNode(x=org[0], y=-org[1], a=-np.radians(org[3]))
    # coordinate of center of mass on target(goal) state, in GCS
    target = OSExplorer.CircleNode(x=aim[0], y=-aim[1], a=-np.radians(aim[3]))
    return source, target


def read_grid(filepath, seq):
    # type: (str, int) -> np.ndarray
    """read occupancy grid map"""
    return cv2.imread(filename='{}/{}_gridmap.png'.format(filepath, seq), flags=-1)


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
    filepath, seq = './test_scenes', 0
    source, target = read_task(filepath, seq)
    # transform coordinate from GCS to LCS.
    start = center2rear(deepcopy(source)).gcs2lcs(source)  # coordinate of rear track center on start state in LCS
    goal = center2rear(deepcopy(target)).gcs2lcs(source)  # coordinate of rear track center on goal state in LCS
    grid_ori = deepcopy(source).gcs2lcs(source)  # coordinate of grid map center in LCS
    grid_map = read_grid(filepath, seq)
    grid_res = 0.1
    explorer = OSExplorer()
    explorer.initialize(start, goal, grid_map=grid_map, grid_res=grid_res, grid_ori=grid_ori)

    def plotter(circles):
        explorer.plot_circles(circles)
        plt.draw()
        raw_input('continue?')
    set_plot(explorer)

    print('Begin?')
    map(explorer.exploring, [None])  # compile jit
    times = 1  # 100
    past = time.time()
    result = map(explorer.exploring, [None]*times)
    now = time.time()
    print('Runtime: {} ms (mean of {} times)'.format(np.round((now - past) / times, 4) * 1000, times))
    print('Done' if sum(result) else 'Find No Path')

    explorer.plot_circles(explorer.circle_path)
    np.savetxt('{}/{}_ose.txt'.format(filepath, seq), explorer.path(), delimiter=',')
    plt.draw()
    raw_input('Plotting')


if __name__ == '__main__':
    main()
