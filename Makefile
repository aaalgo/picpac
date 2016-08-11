CC=g++
CXX=g++
CFLAGS = -g -O3
CXXFLAGS = -fPIC -Ijson11 -ICatch/include -Wall -Wno-sign-compare -std=c++1y -fopenmp -g -O3 -pthread -msse4.2 
#CXXFLAGS += -DSUPPORT_AUDIO_SPECTROGRAM=1
LDFLAGS = -fopenmp
LDLIBS = libpicpac.a $(shell pkg-config --libs opencv) -lboost_timer -lboost_chrono -lboost_program_options -lboost_thread -lboost_filesystem -lboost_system -lglog
SERVER_LIBS = -lserved -lmagic

HEADERS = picpac.h picpac-cv.h picpac-util.h
COMMON = picpac-util.o picpac-cv.o picpac.o json11.o

PROGS = picpac-unpack picpac-crop picpac-split-region picpac-dupe picpac-import-cifar picpac-import-nmist test stress picpac-import picpac-stream picpac-server picpac-proto picpac-roi-scale #picpac-stat picpac-annotate picpac-dumpvec#load-caffe load-dir test test_tr server

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

picpac-server:	picpac-server.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(SERVER_LIBS) $(LDLIBS)
clean:
	rm *.o $(PROGS)

upload_test:
	python setup.py sdist upload -r pypitest

sdist:
	python setup.py sdist 
