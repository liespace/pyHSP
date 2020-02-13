from typing import Any, List
from copy import deepcopy
from explorations import BaseSpaceExplorer
from numba import njit
import numpy as np
import reeds_shepp
import matplotlib.pyplot as plt


class OrientationSpaceExplorer(BaseSpaceExplorer):
    def __init__(self):
        super(OrientationSpaceExplorer, self).__init__()
        self.grid_map = None
        self.grid_res = None
        self.grid_pad = None
        self.obstacle = 255

    def initialize(self, start, goal, **kwargs):
        # type: (BaseSpaceExplorer.CircleNode, BaseSpaceExplorer.CircleNode, Any) -> OrientationSpaceExplorer
        """
        :param start: start circle-node
        :param goal: goal circle-node
        :param kwargs: {grid_map, grid_res}, grid_map: occupancy map(0-1), 2d-square, with a certain resolution: gird_res.
        """
        self.start, self.goal = start, goal
        self.grid_map, self.grid_res = kwargs['grid_map'], kwargs['grid_res']
        # padding grid map for clearance calculation
        s = int(np.ceil((self.maximum_radius + self.minimum_clearance)/self.grid_res))
        self.grid_pad = np.pad(self.grid_map, ((s, s), (s, s)), 'constant',
                               constant_values=((self.obstacle, self.obstacle), (self.obstacle, self.obstacle)))
        # complete the start and goal
        self.start.r, self.start.g = self.clearance(self.start) - self.minimum_clearance, 0
        self.start.h = reeds_shepp.path_length(
            (start.x, start.y, start.a), (self.goal.x, self.goal.y, self.goal.a), 1./self.maximum_curvature)
        self.goal.r, self.goal.h, self.goal.g = self.clearance(self.goal) - self.minimum_clearance, 0, np.inf
        self.start.f, self.goal.f = self.start.g + self.start.h, self.goal.g + self.goal.h
        return self

    def merge(self, expansion, open_set):
        """
        :param expansion: expansion is a set in which items are unordered.
        :param open_set: we define the open set as a set in which items are sorted from Small to Large by cost.
        """
        open_set.extend(expansion)
        open_set.sort(key=lambda item: item.f, reverse=True)

    def pop_top(self, open_set):
        """
        :param open_set: we define the open set as a set in which items are sorted from Small to Large by cost.
        """
        return open_set.pop()

    def exist(self, circle, close_set):
        for item in close_set:
            if self.distance(circle, item) < item.r - self.grid_res:
                return True
        return False

    def overlap(self, circle, goal):
        """
        check if two circles overlap with each other
        in a certain margin (overlap_rate[e.g., 50%] of the radius of the smaller circle),
        which guarantees enough space for a transition motion.
        """
        euler = np.sqrt((circle.x - goal.x)**2 + (circle.y - goal.y)**2)
        r1, r2 = min([circle.r, goal.r]), max([circle.r, goal.r])
        return euler < r1 * self.overlap_rate + r2

    def expand(self, circle):
        def twin(n):
            neighbor = self.CircleNode(x=circle.r * np.cos(n), y=circle.r * np.sin(n), a=n)
            opposite = self.CircleNode(x=circle.r * np.cos(n + np.pi), y=circle.r * np.sin(n + np.pi), a=n)
            neighbor.lcs2gcs(circle)
            opposite.lcs2gcs(circle)
            children.extend([neighbor, opposite])

        def check(child):
            # check if the child is valid, if not, abandon it.
            child.r = min([self.clearance(child) - self.minimum_clearance, self.maximum_radius])
            if child.r > self.minimum_radius:
                # build the child
                child.set_parent(circle)
                # child.h = self.distance(child, self.goal)
                child.h = reeds_shepp.path_length(
                    (child.x, child.y, child.a), (self.goal.x, self.goal.y, self.goal.a), 1. / self.maximum_curvature)
                child.g = circle.g + reeds_shepp.path_length(
                    (circle.x, circle.y, circle.a), (child.x, child.y, child.a), 1. / self.maximum_curvature)
                child.f = child.g + child.h
                # add the child to expansion set
                expansion.append(child)

        children, expansion = [], []
        map(twin, np.radians(np.linspace(-90, 90, self.neighbors / 2)))
        map(check, children)
        # self.plot_circles(children)
        return expansion

    def clearance(self, circle):
        origin, coord = (self.start.x, self.start.y, self.start.a), (circle.x, circle.y, circle.a)
        return self.jit_clearance(coord, origin, self.grid_pad, self.grid_map, self.grid_res,
                                  self.maximum_radius, self.minimum_clearance, self.obstacle)

    @staticmethod
    @njit
    def jit_clearance(coord, origin, grid_pad, grid_map, grid_res, maximum_radius, minimum_clearance, obstacle):
        s_x, s_y, s_a = origin[0], origin[1], origin[2]
        c_x, c_y, c_a = coord[0], coord[1], coord[2]
        x = (c_x - s_x) * np.cos(s_a) + (c_y - s_y) * np.sin(s_a)
        y = -(c_x - s_x) * np.sin(s_a) + (c_y - s_y) * np.cos(s_a)
        u = int(np.floor(y / grid_res + grid_map.shape[0] / 2))
        v = int(np.floor(x / grid_res + grid_map.shape[0] / 2))
        size = int(np.ceil((maximum_radius + minimum_clearance) / grid_res))
        subspace = grid_pad[u:u + 2 * size + 1, v:v + 2 * size + 1]
        rows, cols = np.where(subspace >= obstacle)
        if len(rows):
            row, col = np.fabs(rows - size) - 1, np.fabs(cols - size) - 1
            rs = np.sqrt(row**2 + col**2) * grid_res
            return rs.min()
        else:
            return size * grid_res

    def distance(self, one, another):
        a, b = (one.x, one.y, one.a), (another.x, another.y, another.a)
        return self.jit_distance(a, b, self.maximum_curvature)

    @staticmethod
    @njit
    def jit_distance(one, another, maximum_curvature):
        euler = np.sqrt((one[0] - another[0]) ** 2 + (one[1] - another[1]) ** 2)
        angle = np.abs(one[2] - another[2])
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        # angle = np.pi - angle if angle > np.pi / 2 else angle
        heuristic = angle / maximum_curvature
        return euler if euler > heuristic else heuristic

    def plot_circles(self, circles):
        # type: (List[BaseSpaceExplorer.CircleNode]) -> None
        for circle in circles:
            c = deepcopy(circle)
            c.gcs2lcs(self.start)
            cir = plt.Circle(xy=(c.x, c.y), radius=c.r, color=(0.5, 0.8, 0.5), alpha=0.6)
            arr = plt.arrow(x=c.x, y=c.y, dx=0.5 * np.cos(c.a), dy=0.5 * np.sin(c.a), width=0.1)
            plt.gca().add_patch(cir)
            plt.gca().add_patch(arr)

    @staticmethod
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
