#!/usr/bin/env python
from explorations import BaseSpaceExplorer
from explorers import OrientationSpaceExplorer
from copy import deepcopy
import numpy as np
import cv2
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
    return cv2.imread(filename='{}/{}_gridmap.png'.format(filepath, seq), flags=-1)


def plot_circles(circles):
    for circle in circles:
        circle = plt.Circle(xy=(circle.x, circle.y), radius=circle.r, color=(0.5, 0.8, 0.5), alpha=0.6, lw=0)
        plt.gca().add_artist(circle)


def plot_grid(grid_map, grid_res):
    """plot grid map"""
    row, col = grid_map.shape[0], grid_map.shape[1]
    u = np.array(range(row)).repeat(col)
    v = np.array(range(col) * row)
    uv = np.array([u, v, np.ones_like(u)])
    xy2uv = np.array([[0., 1. / grid_res, row / 2.], [1. / grid_res, 0., col / 2.], [0., 0., 1.]])
    xy = np.dot(np.linalg.inv(xy2uv), uv)
    data = {'x': xy[0, :], 'y': xy[1, :], 'c': np.array(grid_map).flatten() - 1}
    plt.scatter(x='x', y='y', c='c', data=data, s=1., marker="s")


def plot_grid2(grid_map, grid_res):
    """plot grid map"""
    # type: (np.ndarray, float) -> None
    row, col = grid_map.shape[0], grid_map.shape[1]
    indexes = np.argwhere(grid_map == 255)
    xy2uv = np.array([[0., 1. / grid_res, row / 2.], [1. / grid_res, 0., col / 2.], [0., 0., 1.]])
    for index in indexes:
        uv = np.array([index[0], index[1], 1])
        xy = np.dot(np.linalg.inv(xy2uv), uv)
        rect = plt.Rectangle((xy[0] - grid_res, xy[1] - grid_res), grid_res, grid_res, color=(1.0, 0.1, 0.1))
        plt.gca().add_patch(rect)


def main():
    filepath, seq = './test_scenes', 0
    (source, target), (start, goal) = read_task(filepath, seq)
    grid_map = read_grid(filepath, seq)
    grid_res = 0.1
    print (grid_map.min())
    print (source.a, target.a, start.a, goal.a)

    explorer = OrientationSpaceExplorer()
    explorer.initialize(start, goal, grid_map=grid_map, grid_res=grid_res)

    goal.r = explorer.clearance(goal)
    start.r = explorer.clearance(start)
    print (goal.x, goal.y)

    plt.figure()
    plt.gca().set_aspect('equal')
    plt.gca().set_facecolor((0.2, 0.2, 0.2))
    plt.gca().set_xlim((-30, 30))
    plt.gca().set_ylim((-30, 30))
    plot_grid2(grid_map, grid_res)
    plot_circles([goal, start])
    plt.show()


if __name__ == '__main__':
    main()
