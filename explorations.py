from __future__ import print_function
from abc import abstractmethod
from typing import List
import numpy as np


class BaseSpaceExplorer(object):
    def __init__(
            self,
            minimum_radius=0.2,
            maximum_radius=5.0,
            minimum_clearance=1.05,
            neighbors=32,
            maximum_curvature=0.2,
            timeout=1.0,
            overlap_rate=0.5):
        self.minimum_radius = minimum_radius
        self.maximum_radius = maximum_radius
        self.minimum_clearance = minimum_clearance
        self.neighbors = neighbors
        self.maximum_curvature = maximum_curvature
        self.timeout = timeout
        self.overlap_rate = overlap_rate
        self.start, self.goal = None, None

    def initialize(self, start, goal, **kwargs):
        self.start, self.goal = start, goal
        return self

    def exploring(self, plotter=None):
        close_set, open_set = [], [self.start]
        while open_set:
            print (len(open_set))
            circle = self.pop_top(open_set)
            if self.goal.f < circle.f:
                return True
            if not self.exist(circle, close_set):
                self.merge(self.expand(circle), open_set)
                if self.overlap(circle, self.goal) and circle.f < self.goal.g:
                    self.goal.g = circle.f
                    self.goal.set_parent(circle)
            close_set.append(circle)
            if plotter:
                plotter(circle)
        return False

    @property
    def circle_path(self):
        if self.goal:
            path, parent = [self.goal], self.goal.set_parent
            while parent:
                path.append(parent)
                parent = parent.set_parent
            return path
        return []

    @abstractmethod
    def pop_top(self, open_set):
        # type: (List[CircleNode]) -> CircleNode
        """pop the item with the minimum cost (f) of the given set."""
        pass

    @abstractmethod
    def exist(self, circle, close_set):
        # type: (CircleNode, List[CircleNode]) -> bool
        """check if the given circle exists in the given set"""
        pass

    @abstractmethod
    def overlap(self, circle, goal):
        # type: (CircleNode, CircleNode) -> bool
        """check the given two circle are overlapped or not"""
        pass

    @abstractmethod
    def expand(self, circle):
        # type: (CircleNode) -> List[CircleNode]
        """calculate the children of the given circle"""
        pass

    @abstractmethod
    def merge(self, expansion, open_set):
        # type: (List[CircleNode], List[CircleNode]) -> None
        """merge the expanded circle-nodes to the opened set."""
        pass

    @abstractmethod
    def clearance(self, circle):
        # type: (CircleNode) -> float
        """calculate the minimum clearance from the center of the circle-node to the obstacles"""
        pass

    @abstractmethod
    def distance(self, one, another):
        # type: (CircleNode, CircleNode) -> float
        """calculate the distance between two given circle-nodes"""
        pass

    class CircleNode(object):
        def __init__(self, x=None, y=None, a=None, r=None, h=np.inf, g=np.inf, parent=None, children=None):
            self.x = x
            self.y = y
            self.a = a
            self.r = r
            self.h = h  # cost from here to goal, heuristic distance or actual one
            self.g = g  # cost from start to here, actual distance
            self.parent = parent
            self.children = children if children else []

        @property
        def f(self):
            """summed cost of cost h and g"""
            return self.h + self.g

        def set_parent(self, circle):
            self.parent = circle
            circle.children.append(self)

        def lcs2gcs(self, circle):
            # type: (BaseSpaceExplorer.CircleNode) -> None
            """
            transform self's coordinate from local coordinate system (LCS) to global coordinate system (GCS)
            :param circle: the circle-node contains the coordinate (in GCS) of the origin of LCS.
            """
            xo, yo, ao = circle.x, circle.y, circle.a
            x = self.x * np.cos(ao) - self.y * np.sin(ao) + xo
            y = self.x * np.sin(ao) + self.y * np.cos(ao) + yo
            a = self.a + ao
            self.x, self.y, self.a = x, y, a

        def gcs2lcs(self, circle):
            # type: (BaseSpaceExplorer.CircleNode) -> None
            """
            transform self's coordinate from global coordinate system (LCS) to local coordinate system (GCS)
            :param circle: the circle-node contains the coordinate (in GCS) of the origin of LCS.
            """
            xo, yo, ao = circle.x, circle.y, circle.a
            x = (self.x - xo) * np.cos(ao) + (self.y - yo) * np.sin(ao)
            y = -(self.x - xo) * np.sin(ao) + (self.y - yo) * np.cos(ao)
            a = self.a - ao
            self.x, self.y, self.a = x, y, a
