#!/usr/bin/env python3

import csv, json
import drawing
import numpy as np
from util import *
from colors import colors
from sklearn import decomposition

data = load("data.pkl")
log.info(json.dumps(data[0], indent=4))

signals = []
def prep(key):
    log.info("Prepping %s" % key)
    signal = [item[key] for item in data]
    if None in signal:
        log.info("...skipped")
        return
    signal = normalize(signal)
    signal = upsample(signal, 1000 // len(signal))
    signal = smooth(signal, size=50)
    signals.append(signal)

keys = data[0].keys()
for key in keys:
    prep(key)

# ctx = drawing.Context(2000, 800, margin=50, hsv=False)
# for s, signal in enumerate(signals):
#     ctx.plot(signal, stroke=colors[s % len(colors)], thickness=2.0)
# ctx.output("graphs/")

points = np.column_stack(signals)
log.info(points.shape)

print(points[0])
log.debug("INPUT: %s POINTS, %s DIMENSIONS" % points.shape)

points = decomposition.PCA(n_components=8).fit_transform(points)

signals = []
ctx = drawing.Context(2000, 800, margin=50, hsv=False)
for s in range(points.shape[1]):
    signal = normalize(points[:, s])
    signals.append(signal)
    ctx.plot(signal, stroke=colors[s % len(colors)], thickness=2.0)
ctx.output("graphs/")
