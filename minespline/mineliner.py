import collections
import hashlib
import os
import random
import sys
import uuid
from math import hypot, sqrt, exp, pi, tan, radians
from pathlib import Path

import matplotlib
import natsort as natsort
import numba as numba
import numpy
import sklearn.metrics
import geojson as gj
import pandas as pandas
from easydict import EasyDict
from matplotlib import patches
from matplotlib.pylab import plt, Axes
from natsort import natsorted, ns
from numpy import clip, arange, array, zeros
from tqdm import tqdm
import tqdm as tq

import minespline.make_graph as mg
from minespline.config import C
import logging
import time
import click

import pandas as pd
from glob import glob

from multiprocessing import Pool, cpu_count

# noinspection PyBroadException
try:
    # noinspection PyUnresolvedReferences
    profile
except:
    def profile(x):
        return x

logging.basicConfig()
L = logging.getLogger(__name__)


# noinspection PyPep8Naming
def N(x, mu, sigma):
    """Probability density function for the Normal distribution.
    :param x: The variable
    :param mu: Mean
    :param sigma: Standard Deviation
    """
    # Note: We rolled this out ourselves because the scipy version is incredibly slow.
    return exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / sqrt(2 * pi * sigma ** 2)


def geojson2graph(geojson, min_spacing, max_spacing, max_radius, show_progress):
    ds = EasyDict(gj.load(geojson))
    # print(ds)
    points = [p for p in ds.features if p.geometry.type == 'Point']
    xyz, _, ctr = mg.geojson2enu(ds)
    scores = array([p.properties['detection-score'] for p in points])

    # g = mg.make_graph(xyz, scores)
    g = mg.make_graph_hashed(xyz, scores, 0, max_spacing, max_radius, show_progress)
    return g, ds, xyz


import numpy as np


def make_fake_geojson(true_xyz, size, density, ptrue=0.7, pfalse=0.4, pstd=0.2):
    true_xyz[:, 2] = 0  # Z = 0
    min_true = true_xyz.min(0)
    max_true = true_xyz.max(0)
    min_fake = (min_true - size) / 2
    max_fake = (min_true + size) / 2

    cell_size = 1.0 / density
    grid_x, grid_y = numpy.grid[min_fake[0]:max_fake[0]:density, min_fake[1]:max_fake[1]:cell_size]
    grid_x += numpy.random.randint(0, cell_size, grid_x.shape)
    grid_y += numpy.random.randint(0, cell_size, grid_y.shape)

    grid_z = np.zeros_like(grid_x)

    fake_xyz = np.column_stack([grid_x.flat, grid_y.flat, grid_z.flat])

    result = np.row_stack([true_xyz, fake_xyz])

    targets = np.zeros(len(result))
    targets[:len(true_xyz)] = 1

    return result


# Simple structure to hold results of bottom-up processing
PathInfo = collections.namedtuple('PathInfo',
                                  ['path_length',
                                   'max_posterior',
                                   'best_vw',
                                   'best_spacing',
                                   'spacings',
                                   'max_posteriors',
                                   'best_predecessors',
                                   'max_lengths_to_edges'])


class MineLiner:
    def __init__(self,
                 geojson=None,
                 g=None,
                 max_it=None,
                 threshold=None,
                 spacing=None,
                 std_along=None,
                 std_across=None,
                 max_radius=None,
                 prob_miss=None,
                 spacing_from_input=True,
                 min_spacing=4,
                 max_spacing=8,
                 max_angle=10,
                 curve_bias=1,
                 detector_bias=1,
                 step_threshold=0,
                 show_progress=True
                 ):
        """
        Estimate the longest mine line whose posterior exceeds a threshold.

        :param geojson:  A geojson input
        :param g: A graph already constructed from a geojson.
        :param max_it:
        :param threshold:
        :param spacing:
        :param std_along:
        :param std_across:
        :param max_radius:
        :param prob_miss:
        """
        assert geojson is not None or g is not None

        if geojson is not None:
            # The max number of missed mines we handle is 2
            self.g, self.ds, self.xyz = geojson2graph(geojson, min_spacing, max_spacing, max_radius, show_progress)
        else:
            self.g = g

        self.geojson = geojson

        self.std_along = float(std_along if std_along is not None else C.MODEL.STD_ALONG)
        self.std_across = float(std_across if std_across is not None else C.MODEL.STD_ACROSS)
        self.max_radius = float(max_radius if max_radius is not None else C.MODEL.MAX_DISTANCE)
        self.prob_miss = float(prob_miss if prob_miss is not None else C.MODEL.PROB_MISS)
        self.max_it = int(max_it if max_it is not None else sys.maxsize)

        self.threshold = float(threshold if threshold is not None else sys.float_info.epsilon)
        self.step_threshold = step_threshold

        self._spacing = float(spacing if spacing is not None else C.MODEL.SPACING)

        if spacing_from_input:
            self.update_spacing_from_input()

        # These are used if we want to _guess_ the spacing rather than use the spacing provided
        self.guess_spacing = True  # To change this, set the attribute after construction
        self.min_spacing = min_spacing
        self.max_spacing = max_spacing

        # Angle constraint
        self.max_angle = max_angle
        self.tan_angle = tan(radians(max_angle))

        self.curve_bias = curve_bias
        self.detector_bias = detector_bias

        self.p = numpy.array([pv for v, pv in self.g.nodes(data='score')]).clip(0, 1)
        self.P_uv = None
        self.P_w_given_uv = None

        self.alphabeta = None

    def clear_probabilities(self):
        self.P_uv = None
        self.P_w_given_uv = None

    @property
    def spacing(self):
        return self._spacing

    @spacing.setter
    def spacing(self, spacing):
        if spacing != self._spacing:
            self._spacing = spacing
            self.clear_probabilities()

    def update_spacing_from_input(self):
        if 'mine-spacing' in self.ds.properties:
            self.spacing = self.ds.properties['mine-spacing']

    @profile
    def cache_probabilities(self, show_progress=True):
        """Cache the computation of probabilities for each node, edge, and edge-pair
        Profiling has shown these dominated the run time.
        """

        # Reuse the alpha beta cache
        self.cache_alpha_beta(show_progress=show_progress)

        if self.P_uv is None:
            self.P_uv = {(u, v): self.p_uv(u, v)
                         for u, v in tq.tqdm(self.g.edges,
                                             desc="Initialzing P(u->v)",
                                             disable=not show_progress,
                                             leave=False)}

        if self.P_w_given_uv is None:
            self.P_w_given_uv = {}
            for u, v, w in tq.tqdm(self.alphabeta,
                                   desc="Initializing P(v->w | u->v)",
                                   disable=not show_progress,
                                   leave=False):
                alpha, beta, distance, num_missed = self.alphabeta[u, v, w]

                p_uvw = N(alpha, self.spacing, self.std_along) * (1 - self.prob_miss)
                p_uvw += N(alpha / 2, self.spacing, self.std_along) * self.prob_miss
                p_uvw *= N(beta, 0.0, self.std_across * self.spacing)
                pw = self.p[w] ** self.detector_bias
                p_uvw = p_uvw ** self.curve_bias
                p_uvw *= pw
                p_uvw /= p_uvw + ((1 - pw) * 2) / (pi * self.max_radius ** 2) + sys.float_info.epsilon
                self.P_w_given_uv[u, v, w] = p_uvw

    def make_geojson(self, paths):
        """

        :param paths: A list of (path, MAP) pairs in descending order, where each path is a list of points.
        :return:
        """
        # mines = list(self.backtrack(None, PA, L))
        points = [EasyDict(p) for p in self.ds.features if p.geometry.type == 'Point']

        # Update the "point" features
        for i, p in enumerate(points):
            p.properties['marginal'] = 0
            p.properties['predicted'] = 0
            p.properties['curve'] = []
            p.properties['curve-index'] = []
            p.properties['predicted'] = 0

            # NOTE: I wanted to store the 'MAP' path at each
            #       point but I have come to realize this depends
            #       on the length of the path.

        # Create the "edge" line features
        edge_features = {}
        for e in self.g.edges():
            u, v = e
            geom = gj.LineString([points[u].geometry.coordinates, points[v].geometry.coordinates])

            properties = {"line-type": "edge"}
            properties.update(self.g.edges[u, v])
            properties.pop('id')

            f = gj.Feature(id=self.g.edges[e]['id'], geometry=geom, properties=properties)
            edge_features[e] = f

        # Update the "path" features
        curve_features = []
        for i, path_data in enumerate(paths):
            path = path_data[0]
            max_posterior = path_data[1]
            prev_j = None
            for j, cj in enumerate(path):
                # Update the points to indicate that they are part of the curve
                points[cj].properties['curve'].append(i)
                points[cj].properties['curve-index'].append(j)
                points[cj].properties['predicted'] = 1

                # Update the edges to indicate that they are part of the curve
                if prev_j is not None:
                    edge_features[(prev_j, j)].properties['curve'] = i
                    edge_features[(prev_j, j)].properties['curve-index'] = j
                    edge_features[(prev_j, j)].properties['predicted'] = 1

            # Create the "curve" feature
            properties = {"line-type": "curve",
                          "max-posterior": max_posterior,
                          "length": len(path),
                          "nodes": path}
            geom = gj.LineString([points[cj].geometry.coordinates for cj in path])
            curve_features.append(gj.Feature(id=-32930291, geometry=geom, properties=properties))

        # Create a new GeoJSON object and add the features
        ds = gj.FeatureCollection(points + list(edge_features.values()) + curve_features)
        ds['properties'] = dict(self.ds.properties)
        return ds

    def plot_map(self, path_info: PathInfo, title='Length vs MaP'):
        """
        Plot the max-a-posteriori probability of the best mineline for each length from 2...L

        :param path_info: Path information from self.bottom_up()

        :return:
        """
        from matplotlib.pylab import plt

        MAP = path_info.max_posteriors
        L = path_info.path_length

        plt.close(f"MAPs - {id(self)}")
        plt.figure(f"MAPs - {id(self)}")
        plt.title(title)
        plt.yscale('log')
        plt.plot(range(2, L + 1), [MAP[length] for length in range(2, L + 1)])
        plt.annotate(f"Expected = {self.ds.properties['number-mines']}", (self.ds.properties['number-mines'], 0.5),
                     xycoords='data',
                     xytext=(20, 25), textcoords='offset points',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        plt.hlines(self.threshold, 0, L, ls='--')
        plt.vlines(self.ds.properties['number-mines'], 0, 1, ls='--')

        ax = plt.gca()
        minor_ticks = arange(0, L + 1, 1)
        ax.set_xticks(minor_ticks, minor=True)
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

    def plot_path(self, path_info: PathInfo, title: str = 'Paths'):
        """
        Plot the best path identified by 'bottom_up'

        :param path_info: Path information from self.bottom_up()
        :param title: Title of the plot.
        :return:
        """
        from matplotlib.pylab import plt
        path = list(self.backtrack(path_info))
        path = array(path)
        points = [p for p in self.ds.features if p.geometry.type == 'Point']
        expected = array([p.properties.expected for p in points])
        predicted = zeros(len(expected), int)
        predicted[path] = 1

        true_positives = (expected == 1) & (predicted == 1)
        false_positives = (expected == 0) & (predicted == 1)
        false_negatives = (expected == 1) & (predicted == 0)
        true_negatives = (expected == 0) & (predicted == 0)

        plt.close("plot_path - {id(self)}")
        plt.figure("plot_path - {id(self)}", figsize=(10, 10))

        ax = plt.gca()
        plt.plot(self.xyz[path, 0], self.xyz[path, 1], marker='+', c='r')
        plt.scatter(self.xyz[true_negatives, 0], self.xyz[true_negatives, 1], c='gray', alpha=0.5)
        plt.scatter(self.xyz[true_positives, 0], self.xyz[true_positives, 1], c='green', alpha=0.5)
        plt.scatter(self.xyz[false_positives, 0], self.xyz[false_positives, 1], c='blue', alpha=0.5)
        plt.scatter(self.xyz[false_negatives, 0], self.xyz[false_negatives, 1], c='red', alpha=0.5)

        plt.title(title)
        plt.axis('equal')

    def get_point_predictions(self, paths):
        points = [p for p in self.ds.features if p.geometry.type == 'Point']
        expected = [p.properties.expected for p in points]
        predicted = [0] * len(expected)

        for i, (path, max_posterior, best_spacing) in enumerate(paths, start=1):
            for j in path:
                predicted[j] = 1

        return expected, predicted

    def _plot_path_with_arrows(self, path, ax: Axes = None, label=None, **kwargs):
        if ax is None: ax = plt.gca()
        kwargs.update(connectionstyle="arc3,rad=0.2", arrowstyle='<|-', mutation_scale=10)
        for i, j in zip(path[:-1], path[1:]):
            patch = ax.add_patch(patches.FancyArrowPatch(self.xyz[i, :2], self.xyz[j, :2], **kwargs))
        patch.set_label(label)

    def plot_paths(self, paths, title: str = 'Paths', figname=None, figsize=(30, 30)):
        """
        Plot the best path identified by 'bottom_up'

        :param path_info: Path information from self.bottom_up()
        :param title: Title of the plot.
        :return:
        """
        from matplotlib.pylab import plt
        import cycler

        colors = ['r', 'g', 'b', 'y', 'c', 'm', ]

        if figname is None:
            figname = f"plot_path - {id(self)}"
        plt.close(figname)
        fig = plt.figure(figname, figsize=figsize)
        ax = plt.gca()

        points = [p for p in self.ds.features if p.geometry.type == 'Point']
        expected = array([p.properties.expected for p in points])
        predicted = zeros(len(expected), int)

        not_dropped = array([n for n in self.g.nodes])
        dropped = numpy.full(len(points), True)
        dropped[not_dropped] = False

        for i, (path, max_posterior, best_spacing) in enumerate(paths, start=1):
            path = array(path)
            predicted[path] = 1
            # plt.plot(self.xyz[path, 0], self.xyz[path, 1], c='g', alpha=0.5)
            self._plot_path_with_arrows(path, ax, color=colors[(i - 1) % len(colors)], alpha=0.5,
                                        label=f'$|L_{i}|={len(path)},\mu={best_spacing:0.2f}$')

        true_positives = (expected == 1) & (predicted == 1) & ~dropped
        false_positives = (expected == 0) & (predicted == 1) & ~dropped
        false_negatives = (expected == 1) & (predicted == 0) & ~dropped
        true_negatives = (expected == 0) & (predicted == 0) & ~dropped

        plt.scatter(self.xyz[true_negatives, 0], self.xyz[true_negatives, 1], c='gray', alpha=0.5, label='TN')
        plt.scatter(self.xyz[true_positives, 0], self.xyz[true_positives, 1], c='green', alpha=0.5, label='TP')
        plt.scatter(self.xyz[false_positives, 0], self.xyz[false_positives, 1], c='blue', alpha=0.5, label='FP')
        plt.scatter(self.xyz[false_negatives, 0], self.xyz[false_negatives, 1], c='red', alpha=0.5, label='FN')
        plt.scatter(self.xyz[dropped, 0], self.xyz[dropped, 1], c='black', marker='*', s=90, alpha=1.0, label='Dropped')
        plt.legend(loc='upper right')

        plt.title(title)
        plt.autoscale(enable=True, tight=True)
        plt.axis('equal')

        # xmin, xmax = plt.xlim()
        # ymin, ymax = plt.ylim()
        ax = plt.gca()
        # ax.set_xticks(arange(xmin, xmax, self.spacing), minor=True)
        # ax.set_yticks(arange(ymin, ymax, self.spacing), minor=True)
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        # pl.show()
        return fig

    def p_uv(self, u, v):
        """Joint probability of U and V, given their positions"""
        d = self.g.edges[u, v]['length']
        p_uv = N(d, self.spacing, self.std_along) * (1 - self.prob_miss)
        p_uv += N(d / 2, self.spacing, self.std_along) * self.prob_miss
        p_uv = (p_uv ** self.curve_bias) * (self.p[u] * (self.p[v]) ** self.detector_bias)
        denom = p_uv + (1 - self.p[u] * self.p[v]) * 2 * d / (self.max_radius ** 2)
        if denom > 0:
            p_uv /= denom
        else:
            p_uv = 0
        return p_uv

    def p_w_given_uv(self, u, v, w):
        """
        Probability that v->w given u->v, and their positions.

        :param u: Index of vertex u
        :param v: Index of vertex v
        :param w: Index of vertex w
        :return:
        """
        u_xy = self.g.nodes[u]['xy']
        v_xy = self.g.nodes[v]['xy']
        w_xy = self.g.nodes[w]['xy']
        uv = v_xy - u_xy
        vw = w_xy - v_xy
        d_uv = hypot(*uv)

        if d_uv <= 0:
            return 0

        alpha = uv @ vw / d_uv
        beta = sqrt(max(0, vw @ vw - alpha ** 2))

        p_uvw = N(alpha, self.spacing, self.std_along) * (1 - self.prob_miss)
        p_uvw += N(alpha / 2, self.spacing, self.std_along) * self.prob_miss
        p_uvw *= N(beta, 0.0, self.std_across * self.spacing)
        p_uvw = (p_uvw ** self.curve_bias) * (self.p[w] ** self.detector_bias)
        p_uvw /= p_uvw + ((1 - self.p[w]) * 2) / (pi * self.max_radius ** 2) + sys.float_info.epsilon
        # assert p_uvw <= 1
        # assert p_uvw >= 0sys.float_info.epsilon
        return p_uvw

    def bottom_up(self, available_edges=None, show_progress=True):
        if self.guess_spacing:
            info = self._bottom_up_without_spacing(available_edges,
                                                   self.min_spacing, self.max_spacing,
                                                   show_progress=show_progress)
            self.spacing = info.best_spacing
            self.clear_probabilities()
            return self._bottom_up_with_spacing(available_edges,
                                                self.min_spacing, self.max_spacing,
                                                show_progress=show_progress)
        else:
            return self._bottom_up_with_spacing(available_edges, show_progress=show_progress)

    # @profile
    def _bottom_up_with_spacing(self, available_edges=None, min_spacing=None, max_spacing=None, show_progress=True):
        if min_spacing is None:
            min_spacing = self.min_spacing

        if max_spacing is None:
            max_spacing = self.max_spacing

        alphabeta = self.cache_alpha_beta(show_progress=show_progress)

        # Ensure that the conditional probabilities are precomputed in order to improve runtime
        self.cache_probabilities(show_progress=show_progress)

        threshold = self.threshold
        max_it = self.max_it

        # Tracks the MAP of the best path to each edge fora given length
        max_posteriors = {}

        # Records which node precedes each edge on a path
        best_predecessors = {}

        # If no longer path exceeds out threshold, we stop processing an edge
        # and record the length of the best path here.
        max_lengths_to_edges = {}

        # EVEN though we dont use the computed spacings,
        # We still keep track of the best spacing for each path
        spacings = {}

        # At each iteration we potentially remove some edges from consideration.
        # These variables track the previous and next set of edges.
        prev_edges = set(self.g.edges) if available_edges is None else available_edges
        next_edges = set()

        # For length = 2 (each edge is a path)
        path_length = 2
        max_posteriors[path_length] = 0
        best_predecessors[path_length] = 0
        for u, v in tq.tqdm(prev_edges, desc="Initializing Paths", leave=False, disable=not show_progress):
            distance = self.g.edges[u, v]['length']
            if distance < min_spacing:
                # The points are too close -- they cannot be part of the same mineline
                continue

            # It is conceivable that the second, third, etc. point on a line are a false negatives
            # from the detector
            number_of_misses = 0
            while distance / (number_of_misses + 1) > max_spacing:
                number_of_misses += 1

            # The probability of a path of length two
            max_posteriors[u, v, path_length] = self.P_uv[u, v]

            # The vertex that precedes u, v is None, we are a path of length 2
            best_predecessors[u, v, path_length] = None

            # Estimate the spacing of the path based on the first two edges
            spacings[u, v, path_length] = distance / (number_of_misses + 1)

            # Mark the edge as one that can be extended
            next_edges.add((u, v))

            # Keep track of the global "best" path of length 2
            if max_posteriors[u, v, path_length] > max_posteriors[path_length]:
                max_posteriors[path_length] = max_posteriors[u, v, path_length]
                best_predecessors[path_length] = u, v

        # Finished base case, not onto the recursive parts

        progress = tq.tqdm(total=len(prev_edges), desc="Eliminating edges", disable=not show_progress, leave=False)
        for path_length in range(3, max_it):
            progress.update(len(prev_edges) - len(next_edges))

            # We terminate early when no new paths exceed our threshold.
            if len(next_edges) == 0:
                break

            prev_edges = next_edges
            next_edges = set()

            max_posteriors[path_length] = 0
            best_predecessors[path_length] = None
            for (v, w) in tq.tqdm(prev_edges,
                                  desc=f"(L={path_length}) Extending Paths",
                                  leave=False,
                                  disable=not show_progress):
                max_posteriors[v, w, path_length] = 0
                best_predecessors[v, w, path_length] = None

                for u in self.g.predecessors(v):
                    if (u == w) or ((u, v) not in prev_edges):
                        continue

                    cached = alphabeta.get((u, v, w), None)

                    if cached is None:
                        continue

                    alpha, beta, distance, number_of_misses = cached

                    # # Check the angle constraint
                    # if self.tan_angle * alpha < beta:  # angle limit
                    #     continue

                    p_w_uv = self.P_w_given_uv[u, v, w]

                    if p_w_uv < self.step_threshold:
                        continue

                    p_path = max_posteriors[u, v, path_length - 1] * p_w_uv

                    # Estimate the spacing as the average edge length so far
                    # using a recurrence / dp
                    spacing = (1 / (path_length - 1) * (distance / (number_of_misses + 1))
                               + (path_length - 2) / (path_length - 1) * spacings[u, v, path_length - 1])

                    if p_path > max_posteriors[v, w, path_length]:
                        max_posteriors[v, w, path_length] = p_path
                        best_predecessors[v, w, path_length] = u
                        spacings[v, w, path_length] = spacing

                if max_posteriors[v, w, path_length] < threshold:
                    max_lengths_to_edges[v, w] = path_length - 1
                else:
                    next_edges.add((v, w))
                    if max_posteriors[v, w, path_length] > max_posteriors[path_length]:
                        max_posteriors[path_length] = max_posteriors[v, w, path_length]
                        best_predecessors[path_length] = v, w

        progress.close()

        path_length -= 2  # One past the best in the loop
        max_posterior = max_posteriors[path_length]
        best_v, best_w = best_predecessors[path_length]
        spacing = spacings[best_v, best_w, path_length]
        return PathInfo(path_length=path_length,
                        max_posterior=max_posterior,
                        best_spacing=spacing,
                        best_vw=(best_v, best_w),
                        spacings=spacings,
                        max_posteriors=max_posteriors,
                        best_predecessors=best_predecessors,
                        max_lengths_to_edges=max_lengths_to_edges)

    def _bottom_up_without_spacing(self, available_edges=None,
                                   min_spacing=4,
                                   max_spacing=8,
                                   max_it=None,
                                   show_progress=True):
        threshold = self.threshold
        if max_it is None:
            max_it = self.max_it

        alphabeta = self.cache_alpha_beta(show_progress)

        # Tracks the MAP of the best path to each edge fora given length
        max_posteriors = {}

        # Records which node precedes each edge on a path
        best_predecessors = {}

        # If no longer paths through an edge exceed out threshold, we stop
        # processing the edge and record the length of the best path here.
        max_lengths_to_edges = {}

        # We keep track of the best spacing for each path
        spacings = {}

        # At each iteration we potentially remove some edges from consideration.
        # These variables track the previous and next set of edges.
        prev_edges = set(self.g.edges) if available_edges is None else available_edges
        next_edges = set()

        # For length = 2 (each edge is a path)
        path_length = 2
        max_posteriors[path_length] = 0
        best_predecessors[path_length] = 0
        for u, v in tq.tqdm(prev_edges, desc="(Guessing) Initializing Paths", leave=False, disable=not show_progress):

            # For the version of this algorithm where the spacing is unknown
            # we start ignore the spacing for the first two points
            distance = self.g.edges[u, v]['length']
            if distance < min_spacing:
                # The points are too close -- they cannot be part of the same mineline
                continue

            # It is conceivable that the second, third, etc. point on a line are a false negatives
            # from the detector
            number_of_misses = 0
            while distance / (number_of_misses + 1) > max_spacing:
                number_of_misses += 1

            # The probability of a path of length two
            max_posteriors[u, v, path_length] = self.p[u] * self.p[v] * (self.prob_miss ** number_of_misses)

            # The vertex that precedes u, v is None, we are a path of length 2
            best_predecessors[u, v, path_length] = None

            # Estimate the spacing of the path based on the first two edges
            spacings[u, v, path_length] = distance / (number_of_misses + 1)

            # Mark the edge as one that can be extended
            next_edges.add((u, v))

            # Keep track of the global "best" path of length 2
            if max_posteriors[u, v, path_length] > max_posteriors[path_length]:
                max_posteriors[path_length] = max_posteriors[u, v, path_length]
                best_predecessors[path_length] = u, v

        # Finished base case, not onto the recursive parts
        progress = tq.tqdm(total=len(prev_edges), desc="(Guessing) Eliminating edges", disable=not show_progress,
                           leave=False)
        for path_length in range(3, max_it):
            progress.update(len(prev_edges) - len(next_edges))

            # We terminate early when no new paths exceed our threshold.
            if len(next_edges) == 0:
                break

            prev_edges = next_edges
            next_edges = set()

            max_posteriors[path_length] = 0
            best_predecessors[path_length] = None
            best_spacing = 0

            for (v, w) in tqdm(prev_edges, desc=f"(Guessing,L={path_length}) Extending Paths:",
                               disable=not show_progress, leave=False):
                max_posteriors[v, w, path_length] = 0
                best_predecessors[v, w, path_length] = None
                # length_vw = self.g.edges[v, w]['length']

                for u in self.g.predecessors(v):
                    if (u == w) or ((u, v) not in prev_edges):
                        continue


                    cached = alphabeta.get((u, v, w), None)

                    if cached is None:
                        continue

                    alpha, beta, distance, number_of_misses = cached

                    # # Check the angle constraint
                    # if self.tan_angle * alpha < beta:  # angle limit
                    #     continue

                    # Estimate the spacing as the average edge length so far
                    # using a recurrence / dp
                    spacing = (1 / (path_length - 1) * (distance / (number_of_misses + 1))
                               + (path_length - 2) / (path_length - 1) * spacings[u, v, path_length - 1])

                    # This is pretty much a copy of self.p_w_given_uv but we use the computed spacing
                    # and also we have already calculated vectors & points
                    prob_alpha = (1 - self.prob_miss) * N(alpha, spacing, self.std_along)
                    prob_alpha += self.prob_miss * N(alpha / 2, spacing, self.std_along)

                    # This next line could be cached -- not sure if it is worth it
                    prob_beta = N(beta, 0, self.std_across * spacing)
                    pw = self.p[w] ** self.detector_bias
                    p_w_uv = (prob_alpha * prob_beta) ** self.curve_bias * pw
                    p_w_uv /= p_w_uv + ((1 - pw) * 2) / (pi * self.max_radius ** 2) + sys.float_info.epsilon

                    if p_w_uv < self.step_threshold:
                        continue

                    p_path = max_posteriors[u, v, path_length - 1] * p_w_uv

                    if p_path > max_posteriors[v, w, path_length]:
                        max_posteriors[v, w, path_length] = p_path
                        best_predecessors[v, w, path_length] = u
                        spacings[v, w, path_length] = spacing

                if max_posteriors[v, w, path_length] < threshold:
                    max_lengths_to_edges[v, w] = path_length - 1
                else:
                    next_edges.add((v, w))
                    if max_posteriors[v, w, path_length] > max_posteriors[path_length]:
                        max_posteriors[path_length] = max_posteriors[v, w, path_length]
                        best_predecessors[path_length] = v, w
                        best_spacing = spacings[v, w, path_length]

            # tq.tqdm.write(f"Spacing: {best_spacing}");
        progress.close()

        path_length -= 2  # We went one past the end
        max_posterior = max_posteriors[path_length]
        best_v, best_w = best_predecessors[path_length]
        spacing = spacings[best_v, best_w, path_length]
        return PathInfo(path_length=path_length,
                        max_posterior=max_posterior,
                        best_spacing=spacing,
                        best_vw=(best_v, best_w),
                        spacings=spacings,
                        max_posteriors=max_posteriors,
                        best_predecessors=best_predecessors,
                        max_lengths_to_edges=max_lengths_to_edges)

    def cache_alpha_beta(self, show_progress=True):

        edges = self.g.edges

        if self.alphabeta is not None:
            return self.alphabeta

        alphabeta = {}
        for (v, w) in tq.tqdm(edges,
                              leave=False,
                              desc="Caching alpha betas",
                              disable=not show_progress):

            # This is the useful information from the geometry / position of the points vw
            pos_v = self.xyz[v]
            pos_w = self.xyz[w]
            vec_vw = pos_w - pos_v
            length_vw = self.g.edges[v, w]['length']

            if length_vw < self.min_spacing:
                continue

            # It is conceivable that the second, third, etc. point on a line are a false negatives
            # from the detector
            distance = length_vw
            number_of_misses = 0
            while distance / (number_of_misses + 1) > self.max_spacing:
                number_of_misses += 1

            for u in self.g.predecessors(v):
                if (u == w) or ((u, v) not in edges):
                    continue

                # Calculate geometric info for the points u v w
                # (some info is calculated outside the loop already)
                pos_u = self.xyz[u]
                vec_uv = pos_v - pos_u
                length_uv = self.g.edges[u, v]['length']

                if length_uv < self.min_spacing:
                    continue

                # Alpha is the component of the edge vw that is parallel to uv
                alpha = vec_uv @ vec_vw / length_uv

                # Beta is (just the magnitude) of the component perpendicular to uv
                beta = length_vw ** 2 - alpha ** 2
                if beta > 0:
                    beta = sqrt(beta)
                else:
                    beta = 0

                # Check the angle constraint
                if self.tan_angle * alpha < beta:  # angle limit
                    # We do not cache information for edges we prune out because they
                    # use up too much RAM
                    continue

                alphabeta[u, v, w] = alpha, beta, distance, number_of_misses
        self.alphabeta = alphabeta
        return alphabeta

    def backtrack(self, path_info, v=None, w=None):
        """
        Generate the path in reverse order.

        :param path_info:
        :param v:
        :param w:
        :return:
        """
        path_length = path_info.path_length
        best_predecessors = path_info.best_predecessors

        if v is None:
            v, w = best_predecessors[path_length]

        assert w is not None

        yield w
        while path_length > 2 and v is not None:
            u = best_predecessors[v, w, path_length]
            w = v
            v = u
            path_length = path_length - 1
            yield w
        if v is not None:
            yield v

    def find_paths(self, min_length=3, bidir=False, show_progress=True):
        """
        Iterate over all paths whose length exceeds min_length.

        This will not yield overlapping paths -- that is, no two paths will visit the same two
        consecutive nodes in the same order.

        :param min_length: The minimum length of a path to return.
        :param bidir:  Whether to remoe the path AND its reverse
        :param show_progress: Whther to show a progress bar to the console
        :return:
        """
        edges = set(self.g.edges)
        path_info = self.bottom_up(show_progress=show_progress)
        while path_info.path_length >= min_length:
            path = list(self.backtrack(path_info))
            yield path, path_info
            for i in range(len(path) - 1):
                for j in range(i + 1, len(path)):
                    edges.discard((path[j], path[i]))  # We return the path in reverse order...
                    if not bidir:
                        edges.discard((path[i], path[j]))
            path_info = self.bottom_up(edges, show_progress=show_progress)


@click.group(help="General Options")
@click.option('--min-length', type=int,
              help='The minimum length of a path', default=9, show_default=True, show_envvar=True)
@click.option('--top-k', type=int,
              help='The number of paths to return (in descending order)', default=10, show_default=True,
              show_envvar=True)
@click.option('--max-it', '-L', type=int,
              help='The max length of a path', default=sys.maxsize, show_default=True, show_envvar=True)
@click.option('--threshold', '-t', type=float,
              help='The minimum probability of a path', default=1e-35, show_default=True, show_envvar=True)
@click.option('--step-threshold', '-st', type=float,
              help='The minimum reduction in probability of a path at a single step (0 disables)',
              default=0, show_default=True, show_envvar=True)
@click.option('--spacing', '-mu', type=float, default=C.MODEL.SPACING,
              help='The mine spacing, in meters.', show_default=True, show_envvar=True)
@click.option('--std-along', '-sa', type=float, default=C.MODEL.STD_ALONG,
              help='The standard deviation in mine spacing along the mine line, in meters.', show_default=True,
              show_envvar=True)
@click.option('--std-across', '-sx', type=float, default=C.MODEL.STD_ACROSS,
              help='The STD in mine spacing across the mine line, in meters.', show_default=True, show_envvar=True)
@click.option('--curve-bias', type=float, default=1.0,
              help='The priority to put on being a good curve (vs focusing on the probability of each point).',
              show_default=True, show_envvar=True)
@click.option('--detector-bias', type=float, default=1.0,
              help='The priority to put on the detector\s output (vs focusing on a good curve).',
              show_default=True, show_envvar=True)
@click.option('--min-spacing', type=float, default=4,
              help='The minimum distance between two mines', show_default=True, show_envvar=True)
@click.option('--max-spacing', type=float, default=C.MODEL.MAX_DISTANCE,
              help='The maximum distance between two mines', show_default=True, show_envvar=True)
@click.option('--max-angle', type=float, default=10,
              help='The maximum bending angle for a mineline', show_default=True, show_envvar=True)
@click.option('--max-radius', type=float, default='-1',
              help='The maximum search radius. '
                   'If it is -1, we set it to 2*max_spacing+1', show_default=True, show_envvar=True)
@click.option('--prob-miss', type=float, default=C.MODEL.PROB_MISS,
              help='The probability of a false-negative in the input', show_default=True, show_envvar=True)
@click.option('--no-guess-spacing', is_flag=True,
              help="Do not guess the spacing -- instead use the value passed in or specified in the file",
              show_default=True, show_envvar=True)
@click.option('--bi-dir/--single-dir', '-bd/-sd', "bidirectional", is_flag=True, default=True,
              help="Whether to allow the alg. to retrace an edge of a previous path in reverse")
@click.option('--no-progress', is_flag=True, help="Disable progress bars", show_default=True, show_envvar=True)
@click.pass_context
def cli(ctx, **kwargs):
    # Give max radius a meaningful default value in terms of spacing
    if kwargs['max_radius'] < 0:
        kwargs['max_radius'] = 2 * kwargs['max_spacing'] + 1

    ctx.obj.update(**kwargs)


@cli.command(help="Process a single input file and save the results, optionally showing a plot.")
@click.pass_context
@click.argument('input-file', type=click.File('r'))
@click.option('--output', '-o', type=click.File('w'), default=sys.stdout, help='GeoJSON formatted output',
              show_envvar=True)
@click.option('--plot', is_flag=True, help="Display a plot of the results", show_envvar=True)
def show(ctx,
         input_file,
         output,
         plot
         ):
    max_it = ctx.obj['max_it']
    threshold = ctx.obj['threshold']
    step_threshold = ctx.obj['step_threshold']
    spacing = ctx.obj['spacing']
    std_along = ctx.obj['std_along']
    std_across = ctx.obj['std_across']
    max_radius = ctx.obj['max_radius']
    max_spacing = ctx.obj['max_spacing']
    min_spacing = ctx.obj['min_spacing']
    prob_miss = ctx.obj['prob_miss']
    min_length = ctx.obj['min_length']
    top_k = ctx.obj['top_k']
    no_progress = ctx.obj['no_progress']
    no_guess_spacing = ctx.obj['no_guess_spacing']
    curve_bias = ctx.obj['curve_bias']
    detector_bias = ctx.obj['detector_bias']
    max_angle = ctx.obj['max_angle']
    bidirectional = ctx.obj['bidirectional']

    L.setLevel(logging.DEBUG)

    mineliner = MineLiner(input_file,
                          max_it=int(max_it),
                          threshold=float(threshold),
                          step_threshold=float(step_threshold),
                          spacing=float(spacing),
                          std_along=float(std_along),
                          std_across=float(std_across),
                          max_radius=float(max_radius),
                          prob_miss=float(prob_miss),
                          min_spacing=min_spacing,
                          max_spacing=max_spacing,
                          max_angle=max_angle,
                          curve_bias=curve_bias,
                          detector_bias=detector_bias,
                          show_progress=not no_progress
                          )

    if no_guess_spacing:
        mineliner.guess_spacing = False

    L.info('Input has been read and projected to ENU coordinates')

    paths = []
    iterator = mineliner.find_paths(min_length,
                                    bidir=bidirectional,
                                    show_progress=not no_progress)
    iterator = zip(range(top_k), iterator)
    iterator = tqdm(iterator, total=top_k, disable=no_progress)

    for k, (p, info) in iterator:
        paths.append((p, info.max_posterior, info.best_spacing))

    L.info(f"Expected spacing was: {mineliner.spacing}")
    if len(paths) > 0:
        L.info(f"Best spacing was: {paths[0][2]}")

    json = mineliner.make_geojson(paths)
    gj.dump(json, output, indent=True)

    L.info("Saved to output")

    if plot:
        L.info("Plotting")
        mineliner.plot_paths(paths)
        plt.show()


@cli.command(help="Process many files and summarize results, saving a CSV and plots of each file. ")
@click.pass_context
@click.argument('files', type=click.Path(exists=True), nargs=-1)
@click.option('--outdir', '-o', type=click.Path(exists=True, file_okay=False),
              help="Directory to save per-file results", prompt="Output Folder:", show_envvar=True)
@click.option('--p-sep', type=float, help="Set the difference in means (-1 means leave alone) ",
              default=-1, show_default=True, show_envvar=True)
@click.option('--p-std', type=float, help="Set the std. of each class (before clipping to 0/1) (-1 means leave alone) ",
              default=0.2, show_default=True, show_envvar=True)
@click.option('--drop', type=float, help="Fraction of positive examples to drop", default=0, show_default=True,
              show_envvar=True)
@click.option('--processes', '-j', type=int, default=1, show_default=True, show_envvar=True,
              help="The number of assynchronous processes to use. If this is used it limits the number"
                   " of progress bars we can show. A negative value will default to the number of CPU cores.")
@click.option('--rep', type=int, help="Number of times to process each file", default=1, show_default=True,
              show_envvar=True)
@click.option('--index', type=int, show_default=True, show_envvar=True,
              help="Which single file to evaluate, 0 means evaluate only the first file.", default=-1)
@click.option('--plot/--no-plot', default=True, is_flag=True, show_envvar=True, help="Save plots of the input and output of each file")
def evaluate(ctx,
             files,
             outdir,
             p_sep,
             p_std,
             drop,
             processes,
             rep,
             index,
             plot
             ):
    max_it = ctx.obj['max_it']
    threshold = ctx.obj['threshold']
    step_threshold = ctx.obj['step_threshold']
    spacing = ctx.obj['spacing']
    std_along = ctx.obj['std_along']
    std_across = ctx.obj['std_across']
    max_radius = ctx.obj['max_radius']
    max_spacing = ctx.obj['max_spacing']
    min_spacing = ctx.obj['min_spacing']
    prob_miss = ctx.obj['prob_miss']
    min_length = ctx.obj['min_length']
    top_k = ctx.obj['top_k']
    no_progress = ctx.obj['no_progress']
    no_guess_spacing = ctx.obj['no_guess_spacing']
    curve_bias = ctx.obj['curve_bias']
    detector_bias = ctx.obj['detector_bias']
    max_angle = ctx.obj['max_angle']
    bidirectional = ctx.obj['bidirectional']

    ctx.obj['p_sep'] = p_sep
    ctx.obj['p_std'] = p_std
    ctx.obj['drop'] = drop
    ctx.obj['rep'] = rep
    ctx.obj['plot'] = plot

    # Print the config settings out
    for k, v in ctx.obj.items():
        print(f"{k}={v}")

    if processes < 1:
        processes = cpu_count()
    print(f"processes={processes}")

    if (len(files) == 1) and files[0].endswith('.txt'):
        files = [line.strip() for line in open(files[0]).readlines() if line.strip() != ""]

    # Prevent figures from popping up --- this program should render figures to files.
    matplotlib.use('Agg')

    # Collect per-file results to store in a dataframe
    metrics = []

    if index >= 0:
        files = [files[index]]

    # Duplicate the files since we randomly modify parts of them
    # and thus it makes sense to process multiple times
    files *= rep
    files = natsorted(files, alg=ns.IGNORECASE)

    if processes > 1:
        subprocess_no_progress = True
        subprocess_verbose = False
        arguments = [(str(file), index, curve_bias, detector_bias, drop, max_angle, max_it, max_radius, max_spacing,
                      min_length, min_spacing, no_guess_spacing, subprocess_no_progress, outdir, p_sep, p_std,
                      prob_miss,
                      spacing, std_across, std_along, threshold, step_threshold, top_k,
                      subprocess_verbose, plot, bidirectional)
                     for index, file in enumerate(files, start=1)]
        if processes < 1:
            pool = Pool()
        else:
            pool = Pool(processes=processes)
        jobs = pool.imap_unordered(subprocess, arguments)

        print("Using multiple processes to find mines, the first results will not show until after a bit")

        for rec in tqdm(jobs, desc='Overall Progress', total=len(files), disable=no_progress, leave=False):
            i = rec.index
            # Update output
            tq.tqdm.write(f'{i:04} : {rec.Input}')
            row = dict(ctx.obj)
            row.update(rec)
            metrics.append(row)
            metrics.append(rec)
            tq.tqdm.write(str(rec))
    else:
        for i, file in tqdm(enumerate(files, start=1),
                            desc='Overall Progress', total=len(files), disable=no_progress, leave=False):
            tq.tqdm.write(f'{i:04} : {file}')
            rec = _evaluate_single_file(file, i,
                                        curve_bias, detector_bias,
                                        drop, max_angle, max_it, max_radius,
                                        max_spacing,
                                        min_length, min_spacing, no_guess_spacing,
                                        no_progress, outdir, p_sep, p_std,
                                        prob_miss,
                                        spacing, std_across, std_along, threshold,
                                        step_threshold, top_k,
                                        verbose=True, plot=plot,
                                        bidirectional=bidirectional)

            row = dict(ctx.obj)
            row.update(rec)
            # Update output
            metrics.append(row)
            tq.tqdm.write(str(rec))

    metrics = pandas.DataFrame(metrics)

    magic = uuid.uuid4().hex

    metrics.to_csv(os.path.join(outdir, f"eval-{magic}.csv"), header=True, index=False)

    tp = metrics.TP.sum()
    fp = metrics.FP.sum()
    tn = metrics.TN.sum()
    fn = metrics.FN.sum()

    print("----------------")
    print(f"Support: {tp + tn + fp + fn}")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"Precision: {tp / (tp + fp)}")
    print(f"Recall: {tp / (tp + fn)}")
    print(f"F1: {2 * tp / (2 * tp + fp + fn)}")
    print(f"IoU: {tp / (tp + fp + fn)}")


@cli.command(help="Combine CSV files from many runs")
@click.pass_context
@click.argument('csv-files', type=click.Path(exists=True), nargs=-1)
@click.option('--output', '-o',
              type=click.Path(),
              help="CSV File to save per-file results",
              prompt="Output CSV",
              show_envvar=True)
def report(ctx, csv_files, output):
    recs = csv_files
    if output in recs:
        recs.remove(output)
        print(f"Removing output file '{output}' from list of input CSV's.")

    print(f"Concatanating {recs}")
    df = pd.concat([pd.read_csv(rec) for rec in recs])
    df.to_csv(output, index=False)


def subprocess(args):
    return _evaluate_single_file(*args)


def _evaluate_single_file(file, index, curve_bias, detector_bias, drop, max_angle, max_it, max_radius, max_spacing,
                          min_length, min_spacing, no_guess_spacing, no_progress, outdir, p_sep, p_std, prob_miss,
                          spacing, std_across, std_along, threshold, step_threshold, top_k, verbose, plot,
                          bidirectional):
    # Create a mineliner
    start = time.perf_counter()

    with open(file) as input_file:
        L.info(f"Opening {file}")
        mineliner = MineLiner(input_file,
                              max_it=int(max_it),
                              threshold=float(threshold),
                              step_threshold=float(step_threshold),
                              spacing=float(spacing),
                              std_along=float(std_along),
                              std_across=float(std_across),
                              max_radius=float(max_radius),
                              prob_miss=float(prob_miss),
                              min_spacing=min_spacing,
                              max_spacing=max_spacing,
                              max_angle=max_angle,
                              curve_bias=curve_bias,
                              detector_bias=detector_bias,
                              show_progress=not no_progress
                              )
    predictions = array([p.properties.expected
                         for p in mineliner.ds.features if p.geometry.type == 'Point']) == 1
    stem = f"{index:04}-{os.path.splitext(os.path.basename(file))[0]}"

    if drop > 0:
        indices = arange(len(predictions))[predictions]
        drop_indices = numpy.random.choice(indices, round(drop * len(indices)), replace=False)
        for v in drop_indices:
            mineliner.g.remove_node(v)
    if no_guess_spacing:
        mineliner.guess_spacing = False
    if p_sep >= -0.5:
        num_pos = sum(predictions)
        from scipy.stats import norm
        mineliner.p[predictions] = norm.rvs(0.5 + p_sep / 2., p_std, size=num_pos)
        mineliner.p[~predictions] = norm.rvs(0.5 - p_sep / 2., p_std, size=len(mineliner.p) - num_pos)
        mineliner.p = mineliner.p.clip(0.0, 1.0)

    if plot:
        plt.figure('probs', figsize=(10, 5))
        plt.hist(mineliner.p[predictions], bins=20, range=(0, 1), histtype='step', label='pos', density=True)
        plt.hist(mineliner.p[~predictions], bins=20, range=(0, 1), histtype='step', label='neg', density=True)
        plt.legend()
        plt.suptitle(f'{stem}, Histogram of Input P.')
        plt.savefig(os.path.join(outdir, f'{stem}-probs.pdf'))
        plt.close('probs')

    time_init = time.perf_counter() - start
    start = time.perf_counter()

    # Save a plot of just the input
    output_figure_path = os.path.join(outdir, f'{stem}-input.pdf')
    if plot:
        fig = plt.figure(output_figure_path, figsize=(12, 12))
        plt.scatter(mineliner.xyz[:, 0], mineliner.xyz[:, 1], c='gray', alpha=0.5, label='input')
        plt.autoscale(enable=True, tight=True)
        plt.axis('equal')
        plt.suptitle(f'{stem}, Scatter of Input')
        plt.savefig(output_figure_path)
        plt.close(fig.number)
        # sys.exit(0)

    # Save a plot of just the input + probabilities
    output_figure_path = os.path.join(outdir, f'{stem}-input-probs.pdf')
    if plot:
        fig = plt.figure(output_figure_path, figsize=(12, 12))
        # plt.gca().set_facecolor('gray')
        plt.scatter(mineliner.xyz[:, 0], mineliner.xyz[:, 1], c=mineliner.p, cmap=plt.cm.Reds, alpha=1, label='input')
        plt.colorbar()
        plt.autoscale(enable=True, tight=True)
        plt.axis('equal')
        plt.suptitle(f'{stem}, Scatter of Input Prob.')
        plt.savefig(output_figure_path)
        plt.close(fig.number)
        # os.system(f"xdg-open {output_figure_path}"); sys.exit(0)

    # Let the user know the expected number of steps
    if verbose:
        tq.tqdm.write(
            f"{index:06} The number of edges per node (K) is {len(mineliner.g.edges) / len(mineliner.g.nodes)}")
        tq.tqdm.write(f"{index:06} The number of nodes (N) is {len(mineliner.g.nodes)}")
    mineliner.cache_alpha_beta(show_progress=not no_progress)
    if verbose:
        tq.tqdm.write(f"{index:06} Number of triples is {len(mineliner.alphabeta)}")

    # Find the top-k paths
    paths = []
    times = []

    iterator = mineliner.find_paths(min_length,
                                    bidir=bidirectional,
                                    show_progress=not no_progress)
    iterator = zip(range(top_k), iterator)
    iterator = tqdm(iterator,
                    desc='Current File',
                    total=top_k,
                    disable=no_progress,
                    leave=False)

    for k, (p, info) in iterator:
        path_time = time.perf_counter() - start
        start = time.perf_counter()
        paths.append((p, info.max_posterior, info.best_spacing))
        times.append(path_time)
        info: PathInfo

        if verbose:
            tq.tqdm.write(f'{index:06} Found a path L={info.path_length} spacing {info.best_spacing} in {path_time}')

        # Plot and save the plot
        output_figure_path = os.path.join(outdir, f'{stem}-{k + 1:06}.pdf')
        if plot:
            fig = mineliner.plot_paths(paths, figname=output_figure_path, figsize=(12, 12))
            plt.tight_layout()
            plt.suptitle(f'{stem}, Results {k + 1}')
            plt.savefig(output_figure_path)
            plt.close(fig.number)

            # DELETEME
            # os.system(f"xdg-open {output_figure_path}");# sys.exit(0)
    # Save the results
    json = mineliner.make_geojson(paths)
    output_path = os.path.join(outdir, f'results-{stem}.geojson')
    with open(output_path, 'w') as f:
        gj.dump(json, f)
    save_geojson_time = time.perf_counter() - start
    start = time.perf_counter()

    rec = EasyDict()

    expected, predicted = mineliner.get_point_predictions(paths)
    cm = sklearn.metrics.confusion_matrix(expected, predicted)
    tn, fp, fn, tp = cm.ravel()
    rec.index = index
    rec.TP = tp
    rec.TN = tn
    rec.FP = fp
    rec.FN = fn
    rec.Support = tp + tn + fp + fn
    rec.Accuracy = (tp + tn) / max((tp + tn + fp + fn), 1)
    rec.Precision = tp / max((tp + fp), 1)
    rec.Recall = tp / max((tp + fn), 1)
    rec.F1 = 2 * tp / max((2 * tp + fn + fp), 1)
    rec.IoU = tp / max((tp + fn + fp), 1)
    rec.Input = file
    rec.InitTime = time_init
    if len(times) > 0:
        rec.AvePathTime = numpy.mean(times)
    else:
        rec.AvePathTime = 0
    return rec


def evaluate_paths(mineliner, paths, rec):
    return rec


if __name__ == '__main__':
    cli(obj={}, auto_envvar_prefix='ML')  # Should be C?
