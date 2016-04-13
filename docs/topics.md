# Visualizing Annotations

Use the following configuration to generate
annotation images for human inspection:
```python

config.annotate = 'json'
config.anno_copy = True   # use image as background instead of 0s
config.anno_thickness = 1 # default -1 means filling the region
config.anno_color1 = 0xFF # contour color
config.anno_color3 = 0xFF # for color images, channel 3 is read.
                          # for training purpose, default is
                          # anno_color1 = 1, annor_color2/color3=0

config.batch = 1          # one image per batch
config.resize_width = -1  # do not resize (default)
config.resize_height = -1
```

Currently the python API always moves channel axis before rows
and columns, so post processing (e.g. `cv2.merge`) must be done to merge the
channels before images can be saved.



