C++ Server Development Guide
============================

The C++ server has so many dependencies that
a dockerized building environment is needed.

After docker is properly installed, type the
following command to build.

```
./make-server.sh
```

This will download the docker image if it
does not already exist in the system.
It will also pull in all the node.js
dependencies and build the frontend code.

The front-end code building is also lengthy.
If one is paying attention solely to the
C++ part, it is recommened to download a
pre-built front-end packge file
[html_static.o.bz2](http://aaalgo.com/picpac/server/html_static.o.bz2)
and put it in the picpac directory.  When
this file presents, the making process directly
uses this file without building the front-end
from scratch.
