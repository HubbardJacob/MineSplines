import os
import random
import sys

import geojson
import numpy as np
from math import sqrt

from numpy.linalg import norm as length
from scipy.spatial.kdtree import KDTree
from scipy.stats.distributions import norm
import pymap3d

from minespline.config import C
import logging

from minespline.make_graph import geojson2enu

L = logging.getLogger('mine_placement_method')


class MinePlacer(object):
    """ Randomly place a mineline according to a conditional mine placement model.

    This takes an array of `x`, `y`, `z` (optional)  coordinates and associated probabilities `p`
    and it provides a method, `generate` that can generate a sequence of mines.

    The `min_p` parameter is used to decide whether to just invent a new mine instead
    of selecting one from the list of detections. 

    :param p: List of initial detection probabilities
    :param xyz: List of projected (meters) coordinates of initial detections
    :param spacing:  Expected spacing between mines
    :param std_along: Expected STD in spacing along a mine line
    :param std_across: Expected STD in distance across a mine line
    :param min_p: Minimum probability of points in the list of initial detections.

    """

    def __init__(self,
                 p, xyz,
                 spacing=None,
                 std_along=None,
                 std_across=None,
                 max_spacing=None,
                 min_p=None):
        super().__init__()

        assert xyz.shape[1] == 3
        assert len(xyz) == len(p)

        if spacing is None:
            spacing = C.MODEL.SPACING
        spacing = float(spacing)

        if std_along is None:
            std_along = C.MODEL.STD_ALONG
        std_along = float(std_along)

        if std_across is None:
            std_across = C.MODEL.STD_ACROSS
        std_across = float(std_across)

        if max_spacing is None:
            max_spacing = spacing + 4 * sqrt(std_along)  # 99.9% of points would be within 4 std
        max_spacing = float(max_spacing)


        self.xyz = np.array(xyz)
        self.p = np.array(p)
        self.spacing = spacing
        self.std_along = std_along
        self.std_across = std_across
        self.radius = max_spacing
        self.min_p = min_p

        # Probability distributions
        self.prob_along = norm(loc=self.spacing, scale=self.std_along)
        self.prob_across = norm(loc=0, scale=self.std_across)

        # Up-front book keeping for more efficient sampling later
        self.spatial_index = KDTree(self.xyz)
        self.cum_p = np.cumsum(p)
        self.indices = np.arange(len(p))

    def cond_prob(self, xyz, xyz0, xyz1=None):
        """P(xyz | xyz0, xyz1) where xyz0 and xyz1 are the previous two points

        This also works if you only pass in xyz0
        """
        if xyz1 is None:
            xyz0, xyz1 = None, xyz0

        vector = xyz - xyz1
        if xyz0 is None:
            distance_along = length(vector)
            distance_across = 0
        else:
            tangent = (xyz1 - xyz0)
            tangent /= length(tangent)
            normal = np.array([tangent[1], -tangent[0], tangent[2]])
            distance_along = vector @ tangent
            distance_across = vector @ normal

        conditional = self.prob_along.pdf(distance_along) * self.prob_across.pdf(distance_across)
        return conditional

    def cond_rv(self, xyz0, xyz1=None):
        """Sample P(xyz | xyz0, xyz1) to generate a random point that is likely to follow the previous teo

        This also works if you pass in only xyz0
        """
        if xyz1 is None:
            xyz0, xyz1 = None, xyz0

        if xyz0 is None:
            angle = np.random.rand() * 2 * np.pi
            tangent = np.array([np.cos(angle), np.sin(angle), 0])
        else:
            tangent = (xyz1 - xyz0)
            tangent /= length(tangent)
        normal = np.array([tangent[1], -tangent[0], tangent[2]])
        distance_along = self.prob_along.rvs(1)[0]
        distance_across = self.prob_across.rvs(1)[0]

        return xyz1 + distance_along * tangent + distance_across * normal

    def generate(self, p0=None, p1=None, available=None, init_prob=1.0):
        """ Generate an infinitely long sequence of mines.

         When p0 is None
         then a random point is selected according to self.p
         and it will be the first point returned.

         When p0 is not None
         then a random point is selected as p1 according to the model
         and p1 (not p0) will be the first point returned

         When p1 is not None
         then a random point is selected as p2 according to the model
         and p2 is the first point returned.


         The `available` parameter allows us to mask out some of the mines
         for example if they have already been assigned to another
         mineline.

        :param p0:   First point on a mineline (x, y, z)
        :param p1:   Second point on a mineline (x, y, z)
        :param available: array of booleans indicating which mines are available to put on a mineline.
        :param init_prob: The initial probability that there is a mineline anywhere in the dataset.

        :return:  Iterable over pairs (index, xyz, cum_prob),
                  Index may be -1 if this is a pseudo-mine.
                  The last return value (cum_prob) is the probability of the entire path.
        """
        assert p0 is None or len(p0) == 3
        assert p1 is None or len(p1) == 3 and p0 is not None

        cum_prob = float(init_prob) # Probability of the entire path so far (include raw prob * cond. prob)

        prob_along = self.prob_along
        prob_across = self.prob_across

        # Make sure we dont visit the same point twice
        if available is None:
            available = np.full(len(self.p), True)

        # Random choice of the first point, according to prior probability
        if p0 is None:
            i0 = random.choices(population=self.indices, cum_weights=self.cum_p)[0]
            p0 = self.xyz[i0]
            available[i0] = False
            cum_prob += np.log(self.p[i0])
            yield i0, p0, cum_prob

        # Random choice of nth point according to two-point conditional probability
        # This loop will break from the middle when we run out of choices (guaranteed eventually)
        while True:
            if p1 is None:
                # Sample points from spacing+radius because we dont know the direction yet
                neighbors = self.spatial_index.query_ball_point(x=p0, r=self.spacing + self.radius)
                neighbors = np.array(neighbors)
                if len(neighbors) <= 1:
                    use_missing = True
                    missing_mine = self.cond_rv(p0)
                    missing_posterior = self.min_p * self.cond_prob(missing_mine, p0)
                else:
                    neighbors = neighbors[available[neighbors]]  # Eliminates unavailable neighbors
                    priors = self.p[neighbors]
                    vectors = self.xyz[neighbors] - p0

                    distances_along = length(vectors, axis=1)
                    conditionals = prob_along.pdf(distances_along)

                    posteriors = priors * conditionals

                    # No chance to generate a missing point on the first step
                    use_missing = False
                    missing_mine = None
                    missing_posterior = 0
            else:
                tangent = p1 - p0
                tangent /= length(tangent)
                p0, p1 = p1, None

                # Sample points from near where we expect them
                neighbors = self.spatial_index.query_ball_point(x=p0 + tangent * self.spacing, r=self.radius)
                neighbors = np.array(neighbors)
                if len(neighbors) <= 1:
                    use_missing = True
                    missing_mine = self.cond_rv(p0, p1)
                    missing_posterior = self.min_p * self.cond_prob(missing_mine, p0, p1)
                else:
                    neighbors = neighbors[available[neighbors]]  # Eliminates unavailable neighbors
                    priors = self.p[neighbors]
                    vectors = self.xyz[neighbors] - p0

                    # This is the actual distance -- we might want the cosines instead
                    # so that there is a cone instead of a corridor of equally probable points.
                    normal = np.array([tangent[1], -tangent[0], tangent[2]])
                    distances_along = vectors @ tangent
                    distances_across = vectors @ normal
                    conditionals = prob_along.pdf(distances_along) * prob_across.pdf(distances_across)

                    posteriors = priors * conditionals
                    # We also generate a point in case we fail to detect something -- there is
                    # a "background" probability of a mine at any location

                    # This is ccw 90 deg rotation of the tangent -- "left" of the curve
                    missing_along = prob_along.rvs(1)[0]
                    missing_across = prob_across.rvs(1)[0]
                    missing_mine = p0 + missing_along * tangent + missing_across * normal

                    missing_posterior = self.min_p * prob_along.pdf(missing_along) * prob_across.pdf(missing_across)

                    non_missing_weight = sum(posteriors)
                    use_missing = random.random() * (non_missing_weight + missing_posterior) > non_missing_weight

            if use_missing:
                i1 = None
                p1 = missing_mine
                cum_prob += float(np.log(missing_posterior))
            else:
                neighbor_index = random.choices(population=range(len(neighbors)), weights=posteriors)[0]
                i1 = neighbors[neighbor_index]
                p1 = self.xyz[i1]
                cum_prob += float(np.log(posteriors[neighbor_index]))
                available[i1] = False  # We should never visit i1 again

            yield i1, p1, float(cum_prob)


def place_mines(features: geojson.FeatureCollection,
                spacing: float,
                std_along: float,
                std_across: float,
                max_spacing: float,
                min_score: float,
                num_mines: int,
                num_passes: int = 100) -> geojson.FeatureCollection:
    """ Place a sequence of mines

    :param features:
    :param spacing:
    :param std_along:
    :param std_across:
    :param max_spacing:
    :param min_score:
    :return:
    """

    # Get the initial set of points as ECEF coordinates

    enu_xyz, scores,  (lat0, lon0, h0) = geojson2enu(features)

    assert len(enu_xyz) > 0
    assert len(enu_xyz[0]) >= 2
    assert len(scores) == len(enu_xyz)

    placer = MinePlacer(scores,
                        enu_xyz,
                        spacing,
                        std_along,
                        std_across,
                        max_spacing=max_spacing,
                        min_p=min_score)

    all_outputs = []
    for j in range(num_passes):
        outputs=[]
        for i, (output_index, output_point, output_prob) in enumerate(placer.generate()):
            lat, lon, h = pymap3d.enu2geodetic(output_point[0], output_point[1], output_point[2], lat0, lon0, h0)
            # lat, lon, h may be single-element arrays
            lat, lon, h = (float(lat), float(lon), float(h))
            feature = geojson.Feature(geometry=geojson.Point((lat, lon, h)),
                                      properties={'placer-score': output_prob})

            outputs.append(feature)
            if i >= num_mines:
                break
        path_feature = geojson.Feature(geometry=geojson.LineString([ftr.geometry for ftr in outputs]),
                                       properties={'placer-score': outputs[-1].properties['placer-score']})
        all_outputs.append(path_feature)
    result = geojson.FeatureCollection(all_outputs)
    return result


def test_place_mines():
    # Site1 was created with a mine spacing of 30M +/= 5
    points = os.path.join(os.path.dirname(__file__), '..', 'data', 'test', 'site1', 'mines.geojson')
    points = os.path.normpath(points)
    # output = BytesIO()
    with open(points) as f:
        input_features = geojson.load(f)

    output_features = place_mines(features=input_features,
                                  spacing=30,
                                  std_along=5,
                                  std_across=5,
                                  max_spacing=45,
                                  min_score=0.3,
                                  num_mines=10,
                                  num_passes=100
                                  )
    with open(f'test_place_mines.geojson', 'w') as f:
        geojson.dump(output_features, f)



def test_main():
    main(['--points', '../data/test/site1/mines.geojson', '-k', '10'])

import sys

def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--points', type=argparse.FileType('rb'), default=sys.stdin,
                   help='Source of candidate mine locations')
    p.add_argument('--spacing', default=8.,
                   help='The mine spacing, in meters.')
    p.add_argument('--std-along', default=1.,
                   help='The standard deviation in mine spacing along the mine line, in meters.')
    p.add_argument('--std-across', default=1.,
                   help='The STD in mine spacing across the mine line, in meters.')
    p.add_argument('--max-spacing', default=None,
                   help='The maximum distance between two mines')
    p.add_argument('--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
                   help='GeoJSON formatted output')
    p.add_argument('--num', '-n', type=int, default=10,
                   help="The length of the mine line to sample")
    p.add_argument('--min-prob', '-m', type=float, default=0.1,
                   help="Probability of missing a mine in the raw data.")
    p.add_argument('--num-passes', '-k', type=int, default=1,
                   help="Number of mine lines to generate.")

    args = p.parse_args(argv)
    input_features = geojson.load(args.points)
    output_features = place_mines(input_features,
                                  args.spacing,
                                  args.std_along,
                                  args.std_across,
                                  args.max_spacing,
                                  args.min_prob,
                                  args.num,
                                  args.num_passes)
    geojson.dump(output_features, args.output)


if __name__ == '__main__':
    main()
