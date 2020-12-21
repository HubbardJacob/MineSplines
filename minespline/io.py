

import geojson
from easydict import EasyDict

def load(filename):
    with open(filename) as f:
        ds = geojson.load(f)
    ds = EasyDict(ds)

    return ds