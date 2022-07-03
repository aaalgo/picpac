CC=g++
CXX=g++
BUILD_INFO=-DPP_VERSION=\"$(shell git describe --always)\" -DPP_BUILD_ID=\"$(BUILD_ID)\" -DPP_BUILD_NUMBER=\"$(BUILD_NUMBER)\" -DPP_BUILD_TIME=\"$(shell date +%Y-%m-%dT%H:%M:%S)\"
CFLAGS += -g -O3
CXXFLAGS += -fPIC -Ijson11 $(shell pkg-config --cflags opencv4) -I3rd/fmt/include -I3rd/spdlog/include -I3rd/json/single_include/nlohmann -ICatch/include -ISimple-Web-Server -ISimple-Web-Extra -Wall -Wno-sign-compare -std=c++17 -fopenmp -g -O3 -pthread -msse4.2 $(BUILD_INFO)
#CXXFLAGS += -fPIC -Ijson11 $(shell pkg-config --cflags opencv4) -ICatch/include -ISimple-Web-Server -ISimple-Web-Extra -Wall -Wno-sign-compare -std=c++17 -fopenmp -g -O3 -pthread -msse4.2 $(BUILD_INFO)
#CXXFLAGS += -DSUPPORT_AUDIO_SPECTROGRAM=1
LDFLAGS += -fopenmp -std=c++17
LDLIBS = libpicpac.a $(shell pkg-config --libs opencv4) -lboost_timer -lboost_chrono -lboost_program_options -lboost_thread -lboost_system -lglog -lgflags 

SERVER_LIBS = \
          -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_core \
	      -lturbojpeg -ltiff -lpng -lwebp -llibjasper -lippicv \
	      -lboost_timer -lboost_chrono -lboost_program_options -lboost_thread -lboost_filesystem -lboost_system \
	      -lglog -lgflags \
	      -lmagic -lunwind \
	      -lz -lrt -lcares -ldl
 
HEADERS = picpac.h picpac-image.h picpac-util.h
COMMON = picpac-util.o picpac-image.o picpac.o json11.o picpac-image.o shapes.o transforms.o


PROGS = picpac-filter picpac-kfold picpac-unpack picpac-merge picpac-split picpac-downsize picpac-crop picpac-split-region picpac-dupe test stress picpac-import picpac-stream picpac-proto picpac-roi-scale picpac-stat picpac-point2ellipse #picpac-annotate picpac-dumpvec#load-caffe load-dir test test_tr server

.PHONY:	all release python upload_test upload sdist bfdfs

all:	libpicpac.so libpicpac.a $(PROGS) python

python:
	python setup.py build

release:
	rm -rf release
	mkdir -p release
	for p in $(PROGS) ; do cp $$p release/picpac-$$p; done

libpicpac.a:	$(COMMON)
	ar rvs $@ $^

libpicpac.so:	$(COMMON)
	gcc -shared -o $@ $^

%.o:	%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $*.cpp

json11.o:	json11/json11.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $^

$(PROGS):	%:	%.o libpicpac.a
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

picpac-explorer:	picpac-explorer.o html_static.o libpicpac.a
	if [ ! -d /opt/cbox ] ; then echo "!!!BUILD SERVER WITH make-server.sh !!!"; false; fi
	$(CXX) $(LDFLAGS) -static -o $@ $^ $(SERVER_LIBS) 
	mv html_static.o html_static.o.last
	cp $@ $@.full
	objcopy --only-keep-debug $@ $@.debug
	strip -g $@
	cp $@ $@.bin
	upx $@

bfdfs:
	if [ ! -f bfdfs/bfdfs-load ] ; then \
		pushd bfdfs; cmake . ; make ; popd ; fi

html_static.o:	bfdfs
	if [ -f html_static.o.bz2 ]; then	\
		bzcat html_static.o.bz2 > html_static.o ; \
	else \
		make -C copilot ; \
		bfdfs/bfdfs-load $@ copilot/dist --name html_static ; \
	fi
	#cat magic/* > magic.tmp
	#file -C -m magic.tmp
	#mv magic.tmp.mgc html/static/magic.mgc

clean:
	rm *.o $(PROGS)

upload_test:
	python setup.py sdist upload -r pypitest

sdist:
	python setup.py sdist 
