#!/usr/bin/env python3
import sys
import os
import subprocess as sp
HOME = os.path.abspath(os.path.dirname(__file__))
picpac_so = os.path.join(HOME, 'picpac.cpython-35m-x86_64-linux-gnu.so')
# Automatically download picpac library
if not os.path.exists(picpac_so):
    sp.check_call('wget http://www.aaalgo.com/picpac/binary/picpac.cpython-35m-x86_64-linux-gnu.so -O %s' % picpac_so, shell=True)
    pass
import cv2
import simplejson as json
import argparse
import picpac


parser = argparse.ArgumentParser()
# db path
parser.add_argument("--db", default=None, help='')
# output directory
parser.add_argument("--out", default=None, help='')
# file extention that's in the db
parser.add_argument("--ext", default='.jpg', help='')


args = parser.parse_args()

assert args.db, 'please specify --db'
assert args.out, 'please specify --out out_directory'

db = picpac.Reader(args.db)

sp.check_call('mkdir -p %s/images' % args.out, shell=True)
sp.check_call('mkdir -p %s/labels' % args.out, shell=True)
sp.check_call('mkdir -p %s/vis' % args.out, shell=True)

for pk, _, _, fields in db:
    # pk is the serial number of image, integer
    image_path = '%s/images/%06d%s' % (args.out, pk, args.ext)
    label_path = '%s/labels/%06d.txt' % (args.out, pk)
    vis_path = '%s/vis/%06d.jpg' % (args.out, pk)

    image_buf = fields[0]   # this is image buffer
    with open(image_path, 'wb') as f:
        f.write(image_buf)

    # fields[1] is the json annotation
    anno = json.loads(fields[1].decode('ascii'))

    image = cv2.imread(image_path, -1)
    H, W = image.shape[:2]

    with open(label_path, 'w') as f:
        for shape in anno['shapes']:
            assert shape['type'] == 'rect'
            geo = shape['geometry']
            label = shape.get('label', 1)
            x = int(round(geo['x'] * W))
            y = int(round(geo['y'] * H))
            w = int(round(geo['width'] * W))
            h = int(round(geo['height'] * H))
            f.write('%d %g %g %g %g\n' % (label, x, y, w, h))
            cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,255), 1)
        pass
    cv2.imwrite(vis_path, image)
    pass


