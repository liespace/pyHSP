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
            timeout=1.0):
        self.minimum_radius = minimum_radius
        self.maximum_radius = maximum_radius
        self.minimum_clearance = minimum_clearance
        self.neighbors = neighbors
        self.maximum_curvature = maximum_curvature
        self.timeout = timeout
        self.circle_path = None

    def exploring(self, start, goal):
        # type: (CircleNode, CircleNode) -> bool
        closed_set, opened_set = [], [start]
        while opened_set:
            circle = self.pop_top(opened_set)
            if goal.f < circle.f:
                return True
            if not self.exist(circle, closed_set):
                opened_set.extend(self.expand(circle))
                if self.overlap(circle, goal) and circle.f < goal.g:
                    goal.g = circle.f
                    goal.parent(circle)
            closed_set.append(circle)
        return False

    @abstractmethod
    def pop_top(self, opened_set):
        # type: (List[CircleNode]) -> CircleNode
        """pop the item with the minimum cost (f) of the given set."""
        pass

    @abstractmethod
    def exist(self, circle, closed_set):
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
        def __init__(self, x=None, y=None, r=None, h=np.inf, g=np.inf, parent=None, children=None):
            self.x = x
            self.y = y
            self.r = r
            self.h = h  # cost from here to goal, heuristic distance or actual one
            self.g = g  # cost from start to here, actual distance
            self.parent = parent
            self.children = children if children else []

        @property
        def f(self):
            """summed cost of cost h and g"""
            return self.h + self.g

        def parent(self, circle):
            self.parent = circle
            circle.children.append(self)
