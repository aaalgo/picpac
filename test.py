#!/usr/bin/env python3
import sys
import numpy as np
import picpac_ts as pp

writer = pp.Writer("db", pp.OVERWRITE)

for i in range(10):
    writer.append(0, f'{{"range": [{i}, {i+1}]}}',
            np.ones((4,), dtype=np.float32) * i,
            np.ones((4, 3), dtype=np.float32) * i,
            np.ones((4,), dtype=np.float32) * i,
            np.ones((4, 3), dtype=np.float32) * i * -1,
                     )

del writer

config = {
    "db": "db",
    "loop": 0,
    "batch": 1,
    "transforms": [
        #{"type": "regularize", "size": 100, "step": 0.05},
    ]
}

stream = pp.TimeSeriesStream(config)

i = 0
for meta, xxx, data in stream:
    print(i, meta, data.shape)
    print(data)
    i += 1
    if i > 20:
        break
