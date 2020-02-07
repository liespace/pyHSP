from explorations import BaseSpaceExplorer
from typing import Any
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
        start.r, start.h, start.g = self.clearance(start) - self.minimum_clearance, self.distance(start, goal), 0
        goal.r, goal.h, goal.g = self.clearance(goal) - self.minimum_clearance, 0, np.inf
        start.f, goal.f = start.g + start.h, goal.g + goal.h
        return self

    def merge(self, expansion, open_set):
        """
        :param expansion: expansion is a set in which items are unordered.
        :param open_set: we define the open set as a set in which items are sorted from Small to Large by cost.
        """
        open_set.extend(expansion)
        open_set.sort(key=lambda item: item.f)

    def pop_top(self, open_set):
        """
        :param open_set: we define the open set as a set in which items are sorted from Small to Large by cost.
        """
        return open_set.pop(0)

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
        children = []
        for n in np.radians(np.linspace(-90, 90, self.neighbors/2)):
            neighbor = self.CircleNode(x=circle.r * np.cos(n), y=circle.r * np.sin(n), a=n)
            opposite = self.CircleNode(x=circle.r * np.cos(n+np.pi), y=circle.r * np.sin(n+np.pi), a=n)
            neighbor.lcs2gcs(circle)
            opposite.lcs2gcs(circle)
            children.extend([neighbor, opposite])
        expansion = []
        for child in children:
            # check if the child is valid, if not, abandon it.
            child.r = min([self.clearance(child) - self.minimum_clearance, self.maximum_radius])
            if child.r <= self.minimum_radius:
                continue
            # build the child
            child.set_parent(circle)
            child.h = self.distance(child, self.goal)
            child.g = circle.g + reeds_shepp.path_length(
                (circle.x, circle.y, circle.a), (child.x, child.y, child.a), 1./self.maximum_curvature)
            child.f = child.g + child.h
            # add the child to expansion set
            expansion.append(child)
        # self.plot_circles(children)
        return expansion

    def clearance(self, circle):
        s_x, s_y, s_a = self.start.x, self.start.y, self.start.a
        x = (circle.x - s_x) * np.cos(s_a) + (circle.y - s_y) * np.sin(s_a)
        y = -(circle.x - s_x) * np.sin(s_a) + (circle.y - s_y) * np.cos(s_a)
        u = int(np.floor(y/self.grid_res + self.grid_map.shape[0]/2))
        v = int(np.floor(x/self.grid_res + self.grid_map.shape[0]/2))
        size = int(np.ceil((self.maximum_radius + self.minimum_clearance)/self.grid_res))
        subspace = self.grid_pad[u:u+2*size+1, v:v+2*size+1]
        index = np.argwhere(subspace >= self.obstacle)
        if index.shape[0]:
            rs = np.linalg.norm(np.abs(index - size) - 1, axis=-1) * self.grid_res
            return rs.min()
        else:
            return size * self.grid_res

    def distance(self, one, another):
        euler = np.sqrt((one.x - another.x)**2 + (one.y - another.y)**2)
        angle = np.abs(one.a - another.a)
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        angle = np.pi - angle if angle > np.pi/2 else angle
        heuristic = angle / self.maximum_curvature
        return euler if euler > heuristic else heuristic

    @staticmethod
    def plot_circles(circles):
        for circle in circles:
            cir = plt.Circle(xy=(circle.x, circle.y), radius=circle.r, color=(0.5, 0.8, 0.5), alpha=0.6)
            arr = plt.arrow(x=circle.x, y=circle.y, dx=1 * np.cos(circle.a), dy=1 * np.sin(circle.a), width=0.15)
            plt.gca().add_patch(cir)
            plt.gca().add_patch(arr)
