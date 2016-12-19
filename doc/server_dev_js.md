JS Server Development Guide
============================

The JS development code is contained in the
copilot subdirectory.  As many AJAX APIs are
provided by the C++ server, JS cannot be
independently developed.  One needs to
build/download an existing [server binary](http://aaalgo.com/picpac/server/picpac-server)
and some [dataset](http://aaalgo.com/picpac/datasets/)
to assist JS development.

Start the server with the following and keep it running.
```
picpac-server --no-browser any_dataset_file
```
This server will listen at the port 18888.

Then use the regular ```npm run dev``` to develop the
JS part.  C++ API calls are automatically forwarded
to the 18888 port.  The JS frontend will be at the port
8080.
