CC=g++
CXX=g++
CFLAGS = -g -O3
CXXFLAGS = -Wall -Wno-sign-compare -std=c++1y -fopenmp -g -O3 -pthread -msse4.2 
LDFLAGS = -fopenmp
LDLIBS = libpicpac.a -ljson11 $(shell pkg-config --libs opencv) -lboost_timer -lboost_chrono -lboost_program_options -lboost_thread -lboost_filesystem -lboost_system -lglog
SERVER_LIBS = -lserved -lmagic

HEADERS = picpac.h picpac-cv.h picpac-util.h
COMMON = picpac-util.o picpac-cv.o picpac.o

PROGS = test test_tr load-anno server #load-caffe load-dir test test_tr server

.PHONY:	all release

all:	libpicpac.a $(PROGS)

release:
	rm -rf release
	mkdir -p release
	for p in $(PROGS) ; do cp $$p release/picpac-$$p; done

libpicpac.a:	$(COMMON)
	ar rvs $@ $^

%.o:	%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $*.cpp

$(PROGS):	%:	%.o libpicpac.a
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

server:	server.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(SERVER_LIBS) $(LDLIBS)
clean:
	rm *.o $(PROGS)

