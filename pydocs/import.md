# Record Structure

A PicPac database consists a serious of
records, each storing data for a single
training examle.  A record has the
following three fields:

- a float valued label.
- the image.
- (optional) annotation in the form of a
json stream or an image.

The labels returned by the streaming API
are either from the float-valued label
or annotation image pixels, with or without
one-hot encoding depending on the [confuration](config.md)
specified.

### Json Annotation

We support [Annotorious](http://annotorious.github.io/) styple JSON annotation.
Examples are
```javascript
{"shapes":[{"type":"rect","geometry":{"height":0.1329243353783231,"width":0.34801136363636365,"y":0.5766871165644172,"x":0.375}}]}

{"shapes":[{"type":"ellipse","geometry":{"height":0.1329243353783231,"width":0.34801136363636365,"y":0.5766871165644172,"x":0.375}}]}  // ellipse within a bounding rect

{"shapes":[{"type":"polygon","geometry":{"points":[{"y":0.5010660980810234,"x":0.7230113636363636},{"y":0.5010660980810234,"x":0.7230113636363636},{"y":0.4925373134328358,"x":0.796875},{"y":0.5714285714285714,"x":0.7926136363636364},{"y":0.5884861407249466,"x":0.7116477272727273},{"y":0.5010660980810234,"x":0.7159090909090909},{"y":0.4989339019189765,"x":0.7144886363636364},{"y":0.5031982942430704,"x":0.7130681818181818},{"y":0.4989339019189765,"x":0.7130681818181818}]}}]}

```

- Note that all values are relative to width/height, so the same annotation can be applied to resized images. 
- One annotation can contain multiple shapes of same or different types.
- When location/size values are relative to width/height, we cannot easily specify a radius for a circle, so we
choose to support ellipses instead (for now).
- All other fields returned by Annotorious are ignored.

To enable json-style annotation, set config.annotate = "json".  The annotation will then be rendered on-the-fly.


### Image Annotation

We also support image-based annotation, which can be enabled by setting config.annotate = "image".
An training example and it's annotation image must be of the same size.  The annotation image
can only have a single channel.  When annotation pixels are to be interpreted as class IDs,
image must be stored in a lossless format like PNG.

# Import from Commandline
```
$ picpac-import -h
Allowed options:
  -h [ --help ]                  produce help message.
  -i [ --input ] arg
  -o [ --output ] arg
  --max arg (=-1)
  --resize arg (=-1)
  -f [ --format ] arg (=1)
  --cache arg (=".picpac_cache")

Formats:
  0: scan a directory
  1: list of <image	label>
  2: scan a directory
  3: list of <image	json-annotation>
  4: list of <image	annotation-image>
```

We currently support 4 input formats:

- 0: Recursively load all images under directory "input", assigning label 0 to everything.
- 1: From a text file, each line specifying an image path/url and a float label, separated by tab.
- 2: Scan directory "input" for sub-directories named by category IDs 0, 1, ....  Recursively load
each sub-directory and use the category ID as image label.
- 3: From a text file, each line with an image path/url and a json annotation separated by tab.  Assigning label 0 to everything.
- 4: From a text file, each line with an image path/url and a label image path/url, seprated by tab.

When images are given by URL, wget is used to download the image.  The downloaded content is cached
in the specified directory, so when the same URL is imported again the content can be directly loaded from cache.

Although a resize options is provided, it is not recommended.  Use config.resize_width/height for
on-the-fly resizing when streaming.  For images that are really too large, the max option can be used to
limit the image size, so on-the-fly resizing can be more efficient.  The rationale is that the database
does not have to be recreated when training has to be done at different scales.

There is also a Python API for data importing, so the output of cv2.imencode can be directly
added to the database without going through the filesystem.


# Regression vs Classification

Each image can have a float-valued label, which can be interpreted as regression target or
class ID.  In the latter case, the values must be integral.  On-the-fly onehot encoding
can be enabled by setting config.onehot = classes.  When annotations are provided, the
pixels in the annotation image can also be interpreted as regression target of class ID.

