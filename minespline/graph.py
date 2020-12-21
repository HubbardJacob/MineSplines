from typing import Tuple
import networkx as nx
from networkx.readwrite import json_graph
from geojson import FeatureCollection
import matplotlib.pyplot as plt
import json


def make_graph_from_geojson(features: FeatureCollection) -> Tuple[nx.DiGraph, dict]:
    # TODO Connect each pint to the nearby neighboers etc
    g = nx.DiGraph()
    minecount = 0

    for feature in features.features:
        for coordinate in feature["geometry"].coordinates:
            g.add_node(minecount, pos=coordinate)
            minecount += 1

    for x in range(minecount-1):
        g.add_edge(x, x + 1)
        g.add_edge(x + 1, x)

    plt.plot()
    nx.draw_networkx(g, with_labels=True, font_weight='bold')
    plt.show()

    return g, dict()


def demo():
    pass


if __name__ == '__main__':
    demo()
