Imagenet is delivered as a list of tar files.
picpac-import has a special mode to support such format.
To import the full imagenet, generate a list file
containing all the tar paths (using ```find -name '*.tar'```).
The full imagenet is huge and typically images are down-sized
for training.  The recommended setting for importing is below:

```
LD_PRELOAD=./libjpeg.so.8.1.2  ~/picpac/picpac-import -f 6 list imagenet.db
--max 256 --encode .jpg --compact --jpeg_quality 60
```

In the above command the [mozjpeg](https://github.com/mozilla/mozjpeg)
library is used to achieve high compression ratio without breaking
compatibility.  It's also possible to build picpac with a custom-built
OpenCV library with webp support for even better compression ratio.

After importing the database should occupy about 12GB for the
ILSVRC 1000 category dataset, or 150-200GB for the full
20000+ category dataset.

