CC=g++
CXX=g++
BUILD_INFO=-DPP_VERSION=\"$(shell git describe --always)\" -DPP_BUILD_ID=\"$(BUILD_ID)\" -DPP_BUILD_NUMBER=\"$(BUILD_NUMBER)\" -DPP_BUILD_TIME=\"$(shell date +%Y-%m-%dT%H:%M:%S)\"
CFLAGS = -g -O3
CXXFLAGS = -fPIC -Ijson11 -ICatch/include -ISimple-Web-Server -Wall -Wno-sign-compare -std=c++1y -fopenmp -g -O3 -pthread -msse4.2 $(BUILD_INFO)
#CXXFLAGS += -DSUPPORT_AUDIO_SPECTROGRAM=1
LDFLAGS = -fopenmp
LDLIBS = libpicpac.a $(shell pkg-config --libs opencv) -lboost_timer -lboost_chrono -lboost_program_options -lboost_thread -lboost_filesystem -lboost_system -lglog 

SERVER_LIBS = libpicpac.a $(shell pkg-config --libs opencv) \
	      -lboost_timer -lboost_chrono -lboost_program_options -lboost_thread -lboost_filesystem -lboost_system \
	      -lglog -lgflags \
	      -lmagic 

STATIC_SERVER_LIBS = libpicpac.a \
          -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_core -lopencv_hal -lIlmImf -lippicv \
	      -lturbojpeg -ltiff -lpng -ljasper -lwebp \
	      -lboost_timer -lboost_chrono -lboost_program_options -lboost_thread -lboost_filesystem -lboost_system \
	      -lglog -lgflags \
	      -lmagic -lunwind \
	      -lz -lrt -lcares -ldl
 
HEADERS = picpac.h picpac-cv.h picpac-util.h
COMMON = picpac-util.o picpac-cv.o picpac.o json11.o

PROGS = picpac-unpack picpac-merge picpac-split picpac-downsize picpac-crop picpac-split-region picpac-dupe picpac-import-cifar picpac-import-nmist test stress picpac-import picpac-stream picpac-proto picpac-roi-scale picpac-stat #picpac-annotate picpac-dumpvec#load-caffe load-dir test test_tr server

.PHONY:	all release python upload_test upload sdist

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

picpac-server:	picpac-server.o html_static.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(SERVER_LIBS) 
	rm html_static.o 

picpac-server.static:	picpac-server.o html_static.o
	$(CXX) $(LDFLAGS) -static -o $@ $^ $(STATIC_SERVER_LIBS) 
	rm html_static.o 
	objcopy --only-keep-debug $@ $@.debug
	strip -g $@
	cp $@ $@.bin
	upx $@

html_static.o:
	#cat magic/* > magic.tmp
	#file -C -m magic.tmp
	#mv magic.tmp.mgc html/static/magic.mgc
	#make -C copilot
	bfdfs/bfdfs-load $@ copilot/dist --name html_static

clean:
	rm *.o $(PROGS)

upload_test:
	python setup.py sdist upload -r pypitest

sdist:
	python setup.py sdist 
