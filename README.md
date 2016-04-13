PicPac: An Image Streamer for Iterative Training

#[Documentation](http://picpac.readthedocs.org/en/latest/)

#Building

The basic library depends on OpenCV 2.x and Boost.  The dependency on [Json11](https://github.com/dropbox/json11)
is provided as git submodule, which can be pulled in by 
```
git submodule init
git submodule update
```

There's a web server for exploring the database and visualize augmented image/annotations.
The web server depends on [libmagic](https://github.com/threatstack/libmagic) and [served](https://github.com/datasift/served) to build.  The server is not needed for training purpose, but could be useful to make sure the imported
data is correct.

