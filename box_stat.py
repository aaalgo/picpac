#!/usr/bin/env python
import sys
import picpac
import simplejson as json
import cv2
import numpy as np
from sklearn.cluster import KMeans

reader = picpac.Reader(sys.argv[1])
nc = int(sys.argv[2])

sizes = []
for _, _, _, fields in reader:
    if len(fields) == 0:
        continue
    image = fields[0]
    anno = fields[1]
    if len(anno) == 0:
        continue
    image = cv2.imdecode(np.fromstring(image, np.uint8), cv2.CV_LOAD_IMAGE_COLOR)
    H, W = image.shape[0:2]
    anno = json.loads(anno)
    for box in anno['shapes']:
        assert box['type'] == 'rect'
        geo = box['geometry']
        _, _, width, height = geo['x'], geo['y'], geo['width'], geo['height']
        width *= W
        height *= H
        sizes.append([width, height])
        pass
array = np.array(sizes, dtype=np.float32)
print array.shape
kmeans = KMeans(n_clusters=nc, random_state=1).fit(array)
print kmeans.cluster_centers_




