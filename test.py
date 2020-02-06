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
        cir = plt.Circle(xy=(circle.x, circle.y), radius=circle.r, color=(0.5, 0.8, 0.5), alpha=0.6)
        arr = plt.arrow(x=circle.x, y=circle.y, dx=1*np.cos(circle.a), dy=1*np.sin(circle.a), width=0.15)
        plt.gca().add_patch(cir)
        plt.gca().add_patch(arr)


def plot_grid2(grid_map, grid_res):
    # type: (np.ndarray, float) -> None
    """plot grid map"""
    row, col = grid_map.shape[0], grid_map.shape[1]
    u = np.array(range(row)).repeat(col)
    v = np.array(range(col) * row)
    uv = np.array([u, v, np.ones_like(u)])
    xy2uv = np.array([[0., 1. / grid_res, row / 2.], [1. / grid_res, 0., col / 2.], [0., 0., 1.]])
    xy = np.dot(np.linalg.inv(xy2uv), uv)
    data = {'x': xy[0, :], 'y': xy[1, :], 'c': np.array(grid_map).flatten() - 1}
    plt.scatter(x='x', y='y', c='c', data=data, s=1., marker="s")


def plot_grid(grid_map, grid_res):
    # type: (np.ndarray, float) -> None
    """plot grid map"""
    row, col = grid_map.shape[0], grid_map.shape[1]
    indexes = np.argwhere(grid_map == 255)
    xy2uv = np.array([[0., 1. / grid_res, row / 2.], [1. / grid_res, 0., col / 2.], [0., 0., 1.]])
    for index in indexes:
        uv = np.array([index[0], index[1], 1])
        xy = np.dot(np.linalg.inv(xy2uv), uv)
        rect = plt.Rectangle((xy[0] - grid_res, xy[1] - grid_res), grid_res, grid_res, color=(1.0, 0.1, 0.1))
        plt.gca().add_patch(rect)


def set_plot():
    plt.ion()
    plt.figure()
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().set_aspect('equal')
    plt.gca().set_facecolor((0.2, 0.2, 0.2))
    plt.gca().set_xlim((-30, 30))
    plt.gca().set_ylim((-30, 30))


def main():
    filepath, seq = './test_scenes', 0
    (source, target), (start, goal) = read_task(filepath, seq)
    grid_map = read_grid(filepath, seq)
    grid_res = 0.1
    explorer = OrientationSpaceExplorer()
    explorer.initialize(start, goal, grid_map=grid_map, grid_res=grid_res)

    set_plot()
    plot_grid(grid_map, grid_res)
    plt.draw()
    raw_input('continue?')

    def plotter(circle):
        plot_circles([circle])
        plt.draw()
        raw_input('continue?')

    if explorer.exploring(plotter=plotter):
        circle_path = explorer.circle_path
        plot_circles(circle_path)
        plt.show()
    else:
        print ('No Path!!!')


if __name__ == '__main__':
    main()
