from explorations import BaseSpaceExplorer
from typing import Any
import numpy as np
import reeds_shepp


class OrientationSpaceExplorer(BaseSpaceExplorer):
    def __init__(self):
        super(OrientationSpaceExplorer, self).__init__()
        self.grid_map = None
        self.grid_res = None
        self.grid_pad = None

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
        self.grid_pad = np.pad(self.grid_map, ((s, s), (s, s)), 'constant', constant_values=((1, 1), (1, 1)))
        return self

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
        children = []
        for f in np.radians(np.linspace(-90, 90, self.neighbors/2)):
            neighbor = self.CircleNode(x=circle.r * np.cos(f), y=circle.r * np.sin(f), a=f)
            opposite = self.CircleNode(x=circle.r * np.cos(f+np.pi), y=circle.r * np.sin(f+np.pi), a=f)
            neighbor.transform(circle)
            opposite.transform(circle)
            children.extend([neighbor, opposite])

        expansion = []
        for child in children:
            # check if the child is valid, if not, abandon it.
            child.r = min([self.clearance(child) - self.minimum_clearance, self.maximum_radius])
            if child.r <= self.minimum_radius:
                continue
            # build the child
            child.parent(circle)
            child.h = self.distance(child, self.goal)
            child.g = circle.g + reeds_shepp.path_length(
                (circle.x, circle.y, circle.a), (child.x, child.y, child.a), 1./self.maximum_curvature)
            # add the child to expansion set
            expansion.append(child)

    def clearance(self, circle):
        s_x, s_y, s_a = self.start.x, self.start.y, self.start.a
        x = (circle.x - s_x) * np.cos(s_a) + (circle.y - s_y) * np.sin(s_a)
        y = -(circle.x - s_x) * np.sin(s_a) + (circle.y - s_y) * np.cos(s_a)
        u = int(np.floor(y/self.grid_res + self.grid_map.shape[0]/2))
        v = int(np.floor(x/self.grid_res + self.grid_map.shape[0]/2))

        size = int(np.ceil((self.maximum_radius + self.minimum_clearance)/self.grid_res))
        subspace = self.grid_pad[u:u+2*size+1, v:v+2*size+1]

        r = size * self.grid_res
        for i in range(1, size+1):
            u0, u1 = size - i, size + i
            v0, v1 = size - i, size + i
            rs = []
            us1, vs1 = range(u0, u1+1)*2, [v0]*(2*i+1) + [v1] * (2*i+1)
            us2, vs2 = [u0]*(2*i+1) + [u1] * (2*i+1), range(v0, v1+1) * 2
            indexes = np.transpose(np.array([us1 + us2, vs1 + vs2]))
            for index in indexes:
                if subspace[index[0], index[1]] > 0:
                    rs.append(np.linalg.norm([(size - u) * self.grid_res, (size - v0) * self.grid_res]))
            if rs:
                r = min(rs)
                break
        return r

    def distance(self, one, another):
        euler = np.linalg.norm([one.x - another.x, one.y - another.y])
        heuristic = np.abs(one.a - another.a) / self.maximum_curvature
        return max([euler, heuristic])
