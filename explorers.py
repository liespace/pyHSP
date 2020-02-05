from explorations import BaseSpaceExplorer
import numpy as np


class OrientationSpaceExplorer(BaseSpaceExplorer):
    def merge(self, expansion, open_set):
        """
        :param expansion: expansion is a set in which items are unordered.
        :param open_set: we define the open set as a set in which items are sorted from Small to Large by cost.
        """
        for exp in expansion:
            # find the index to insert.
            index = 0
            for item in open_set:
                if exp.f < item.f:
                    break
                index += 1
            # insert to the open-set.
            open_set.insert(index, exp)

    def pop_top(self, open_set):
        """
        :param open_set: we define the open set as a set in which items are sorted from Small to Large by cost.
        """
        return open_set[0]

    def exist(self, circle, close_set):
        for item in close_set:
            if self.distance(circle, item) <= item.r:
                return True
        return False

    def overlap(self, circle, goal):
        """
        check if two circles overlap with each other
        in a certain margin (overlap_rate[e.g., 50%] of the radius of the smaller circle),
        which guarantees enough space for a transition motion.
        """
        euler = np.linalg.norm([circle.x - goal.x, circle.y - goal.y])
        r1, r2 = min([circle.r, goal.r]), max([circle.r, goal.r])
        return euler < r1 * self.overlap_rate + r2

    def expand(self, circle):
        pass

    def clearance(self, circle):
        pass

    def distance(self, one, another):
        euler = np.linalg.norm([one.x - another.x, one.y - another.y])
        heuristic = np.abs(one.a - another.a) / self.maximum_curvature
        return max([euler, heuristic])
