CC=g++
CXX=g++
CFLAGS = -g -O3
CXXFLAGS = -Ijson11 -Wall -Wno-sign-compare -std=c++1y -fopenmp -g -O3 -pthread -msse4.2 
LDFLAGS = -fopenmp
LDLIBS = libpicpac.a $(shell pkg-config --libs opencv) -lboost_timer -lboost_chrono -lboost_program_options -lboost_thread -lboost_filesystem -lboost_system -lglog
SERVER_LIBS = -lserved -lmagic

HEADERS = picpac.h picpac-cv.h picpac-util.h
COMMON = picpac-util.o picpac-cv.o picpac.o json11.o

PROGS = stress test test_tr load-anno server #load-caffe load-dir test test_tr server

.PHONY:	all release python upload_test upload sdist

all:	libpicpac.a $(PROGS) python

python:
	python setup.py build

release:
	rm -rf release
	mkdir -p release
	for p in $(PROGS) ; do cp $$p release/picpac-$$p; done

libpicpac.a:	$(COMMON)
	ar rvs $@ $^

%.o:	%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $*.cpp

json11.o:	json11/json11.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $^

$(PROGS):	%:	%.o libpicpac.a
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

server:	server.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(SERVER_LIBS) $(LDLIBS)
clean:
	rm *.o $(PROGS)

upload_test:
	python setup.py sdist upload -r pypitest

sdist:
	python setup.py sdist 
