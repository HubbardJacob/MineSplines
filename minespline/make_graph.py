import io
from random import shuffle

import geojson
import numpy as np
import pymap3d
from easydict import EasyDict
from scipy.spatial.distance import squareform, pdist
from scipy.stats import norm

from minespline.config import C
import networkx as nx
import tqdm as tq

import math


def geojson2enu(features):
    """
    Load ENU coordinates and scores from a geojson file.

    :param features:  A FeatureCollection or the name of a geojson
    :return:
    """
    if not isinstance(features, (geojson.FeatureCollection, dict)):
        if isinstance(features, io.IOBase):
            features = geojson.load(features)
        else:
            with open(features) as f:
                features = geojson.load(f)

    if not isinstance(features, EasyDict):
        features = EasyDict(features)

    ecef_xyz = []
    scores = []
    for feature in features.features:
        if feature.geometry.type == 'Point':
            llh = feature.geometry.coordinates
            if len(llh) == 2:
                llh.append(0)  # altitude
            lon, lat, h = llh
            x, y, z = pymap3d.ecef.geodetic2ecef(lat, lon, h)
            score = feature.properties.get('detection-score', 0.5)
            ecef_xyz.append((x, y, z))
            scores.append(score)
    ecef_xyz = np.array(ecef_xyz)
    # Convert to ENU coordinates
    lat0, lon0, h0 = pymap3d.ecef.ecef2geodetic(*ecef_xyz[:, :3].mean(0))
    enu_xyz = pymap3d.ecef.ecef2enu(ecef_xyz[:, 0], ecef_xyz[:, 1], ecef_xyz[:, 2], lat0, lon0, h0)
    enu_xyz = np.column_stack(enu_xyz)
    assert len(enu_xyz) > 0
    assert len(enu_xyz[0]) >= 2
    assert len(scores) == len(enu_xyz)
    return enu_xyz, scores, (lat0, lon0, h0)


def score_dist(d, spacing, std_along, prob_miss):
    result = norm.pdf(d - spacing, scale=std_along)
    result += prob_miss * norm.pdf(d - 2 * spacing, scale=2 * std_along)
    return result


def plot_score_dist(spacing, std_along, prob_miss, max_distance):
    from matplotlib.pylab import plt
    plt.close("Score Dist")
    plt.figure("Score Dist")
    d = np.linspace(0, max_distance, 500)
    plt.plot(d, [score_dist(di, spacing, std_along, prob_miss) for di in d])
    plt.vlines(spacing, 0, 1)
    plt.vlines(spacing * 2, 0, 1, ls='--')
    plt.annotate("Miss-detect the next mine", (spacing * 2, 0.5), (12, 0), textcoords='offset points')
    plt.ylabel('$p(d)$')
    plt.xlabel('$d$')
    plt.grid()
    plt.xticks(np.arange(max_distance))
    plt.xlim(0, max_distance)
    plt.savefig('score_dist.pdf')


def make_graph(xyz, score, spacing, std_along, prob_miss, max_distance, show_progress=True):
    # TODO:  Do this directly from the geojson -- include all properties in the graph
    g = nx.DiGraph()
    d = squareform(pdist(xyz))
    for v, point in enumerate(xyz):
        g.add_node(v, xyz=point, xy=point[:2], score=score[v])
    # TODO:  Use a kdtree or hash....
    eid = 0
    for u, p1 in tq.tqdm(enumerate(xyz), total=len(xyz), desc="Building graph", disable=not show_progress, leave=False):
        for v, p2 in enumerate(xyz):
            if (u != v) and (d[u, v] < max_distance):
                g.add_edge(u, v, id=eid)
                eid += 1
                g.edges[u, v].update({
                    'length': d[u, v],
                    'length-score': score_dist(d[u, v], spacing, std_along, prob_miss),
                    'max-posterior': 0,
                    'max-predecessor': 's',
                    'marginal': 0,
                    "threat_class": 'NOT SET',
                    "threat_type": 'NOT SET',
                    "placement": 'NOT SET',
                    "burial": 'NOT SET',
                })
    return g


def make_graph_hashed(xyz, score,
                      min_spacing,
                      max_spacing,
                      max_radius,
                      show_progress=True):
    # TODO:  Do this directly from the geojson -- include all properties in the graph
    g = nx.DiGraph()
    d = squareform(pdist(xyz))
    for v, point in enumerate(xyz):
        g.add_node(v, xyz=point, xy=point[:2], score=score[v])

    ixy = (xyz[:, :2] / max_radius).round().astype(int)

    hash = {}
    for (x, y) in ixy:
            hash[x, y] = []

    for i, (x, y) in enumerate(ixy):
        hash[x, y].append(i)

    eid = 0
    for u, (ux, uy) in tq.tqdm(enumerate(ixy), total=len(xyz), desc="Building graph", disable=not show_progress, leave=False):
        for vx in ux-1, ux, ux+1:
            for vy in uy-1, uy, uy+1:
                for v in hash.get((vx, vy), []):
                    if (u != v) and (min_spacing <= d[u, v] < max_radius):
                        g.add_edge(u, v, id=eid)
                        eid += 1
                        g.edges[u, v].update({
                            'length': d[u, v],
                            'max-posterior': 0,
                            'max-predecessor': 's',
                            'marginal': 0,
                            "threat_class": 'NOT SET',
                            "threat_type": 'NOT SET',
                            "placement": 'NOT SET',
                            "burial": 'NOT SET',
                        })
    return g


def plot_graph(g, ax=None, with_labels=True):
    from matplotlib.pylab import plt
    if ax is None:
        ax = plt.gca()
    ax.set_facecolor('lightgray')
    edges = np.array(g.edges())
    edge_scores = np.array([a['score'] for a in g.edges().values()])
    # edge_scores = np.log(edge_scores)
    # edge_scores += edge_scores.min()
    order = np.argsort(edge_scores)

    nx.draw_networkx_nodes(g, nx.get_node_attributes(g, 'xy'),
                           ax=ax,
                           node_size=30,
                           font_size=9,
                           font_color='blue',
                           with_labels=with_labels,
                           )
    edges = nx.draw_networkx_edges(g, nx.get_node_attributes(g, 'xy'),
                                   ax=ax,
                                   edgelist=edges[order].tolist(),
                                   edge_color=edge_scores[order],
                                   edge_cmap=plt.cm.gray_r,
                                   node_size=30,
                                   edge_vmin=edge_scores[order[0]],
                                   edge_vmax=edge_scores[order[-1]],
                                   font_size=9,
                                   font_color='blue',
                                   with_labels=with_labels,
                                   connectionstyle='arc3,rad=0.2'
                                   )

    ax.set_aspect('equal')

    return edges


def make_cpt(g: nx.DiGraph, mu_along, std_along, std_across, prob_miss, max_distance):
    """
    Make the conditional probability table (cpt) of the graph.

    This adds a source ('s') and sink ('t') node to the graph.

    The cpt[(u,v), (v,w)] is the probability of (v,w) given (u,v)

    :param g:
    :param mu_along:
    :param std_along:
    :param std_across:
    :param prob_miss:
    :return: G, cpt
    """
    g = g.copy()

    # Add start and end 'pseudo' edges to the graph
    #     s = G.add_node('s', score=1.)
    #     t = G.add_node('t', score=1.)
    g.add_node('s', score=1.)
    g.add_node('t', score=1.)

    start_penalty = 0.1
    end_penalty = 0.1
    for v in g.nodes():
        if v not in ('s', 't'):
            g.add_edge('s', v, score=start_penalty)
            g.add_edge(v, 't', score=end_penalty)

    edges = g.edges
    node_pos = nx.get_node_attributes(g, 'xy')
    node_score = nx.get_node_attributes(g, 'score')

    cpt = {}
    for (u, v) in edges:
        for _, w in g.out_edges(v):
            if w != u:
                # Then it is an out edge that is not starting at the end node,
                # nor is it the reverse direction of the current edge
                if w == 't':
                    # The last edge in a mine line so we have no 'next' position
                    score = end_penalty
                elif u == 's':
                    # The first edge in a mine line
                    # so we do not have a "current" direction (s has no position)
                    vw = node_pos[w] - node_pos[v]
                    dist_vw = np.linalg.norm(vw)
                    dist_along = dist_vw
                    score = (score_dist(dist_along, mu_along, std_along, prob_miss) * node_score[w] * node_score[v])
                    score = score / (score + (
                            1 - node_score[w] * node_score[v]) * dist_along * 2 / max_distance ** 2) * start_penalty
                else:
                    # Use the distance along vs across the current line
                    uv = node_pos[v] - node_pos[u]
                    uw = node_pos[w] - node_pos[u]
                    dist_uv = np.linalg.norm(uv)
                    dist_uw = np.linalg.norm(uw)
                    dist_along = uv @ uw / dist_uv
                    c = dist_along / dist_uw
                    s = np.sqrt(1 - c ** 2)
                    dist_across = dist_uw * s

                    score = (score_dist(dist_along, mu_along, std_along, prob_miss)
                             * norm.pdf(dist_across, 0, std_across)
                             * node_score[w])

                if score < float('1.0e-5'):
                    score = 0.0

                cpt[(u, v), (v, w)] = score

    return g, cpt


def edge_probabilities(g, cpt, threshold=False):
    # Given cpt and the graph:
    # P(x_1~x_2) = sum_{parents}{P(x_p~x_1, x_1~x_2)}
    # For the CPT we have the keys as ((x_p~x_1), (x_1~x_2)) : P((x_1~x_2) | (x_p~x_1))
    # Thus all the parents are all key[0] where key[1] == (x_1~x_2)

    g = g.copy()

    if threshold:
        ept = {(u, v): max(edge_score, float('1e-5')) for u, v, edge_score in g.edges(data='score')}
        # Some incoming values were very close to the smallest possible float
        # which would get converted to nan
    else:
        ept = {(u, v): edge_score for u, v, edge_score in g.edges(data='score')}
        # Initialize a dict of the form (u~v) : p(u~v) with the scores

    for edge in nx.edge_bfs(g, 's'):
        # Breadth first search starting at s for efficiency
        ept[edge] = edge_probability(cpt, edge, ept, 0)
        # Find P(edge), gets recursive Whoa!!!

    if threshold:
        # May want to remove values below a certain cutoff
        ept = {key: max(value, float('1e-5')) for key, value in ept.items()}

    return g, cpt, ept


def edge_probability(cpt, edge, ept, current_depth, max_depth=2):
    if (edge[0] == 's') or (edge[1] == 't'):
        # The edge is from the start node to another node or
        # from another node to the end node
        return ept[edge]
    elif current_depth == max_depth:
        # Stop looping, going to hit the original edge, use the initial value
        return ept[edge]
    else:
        # Recursively find probabilities of parents and return this one
        parent_cpts = {key: value for key, value in cpt.items() if key[1] == edge}

        edge_prob = 0.0
        for key, value in parent_cpts.items():
            # Value is the conditional of the edge given the parent
            parent = key[0]

            parent_probability = edge_probability(cpt, parent, ept, current_depth + 1)
            if math.isnan(parent_probability):
                # Might be numerical or depth related, but im tired
                # TODO: investigate:  parent_probability = ept[parent]
                parent_probability = 0.0

            ept[parent] = parent_probability
            # Store for future use
            edge_prob += value * parent_probability
        return edge_prob


class DefaultCheck:
    def __init__(self, graph):
        self.graph = graph

    def __call__(self, k, err):
        done = False

        print(f" [{k:06d}] err={err}")
        return done


def max_sum(g: nx.DiGraph, cpt, s, t, start_penalty, end_penalty, max_iterations, check):
    edges = list(g.edges)

    node_p = {}
    edge_p = {}

    for v in g.nodes:
        if v is s or v is t:
            continue
        node_p[v] = g.nodes[v]['score']
        edge_p[(s, v)] = start_penalty * node_p[v]
        edge_p[(v, t)] = start_penalty * node_p[v] * end_penalty

    for u, v in edges:
        if u is s or v is t:
            continue
        edge_p[(u, v)] = 0

    for k in range(max_iterations):
        delta = 0
        shuffle(edges)
        for v, w in edges:
            if v == s:
                continue

            p = 0
            sum_cpt = 0
            for u, _ in g.in_edges(v):
                p += edge_p[(u, v)] * cpt[(u, v), (v, w)]
                sum_cpt += cpt[(u, v), (v, w)]
            p /= sum_cpt
            delta += abs(node_p[v] - p)
            node_p[v] = p

        if check(k, delta):
            break
