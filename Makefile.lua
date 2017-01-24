CXXFLAGS_EXTRA = -Ijson11 -Wall -Wno-sign-compare -std=c++1y -g -O3 -pthread 
LDLIBS = -lopencv_highgui -lopencv_core -lboost_filesystem -lboost_system -lboost_python -lglog -lluaT -lTH -lluajit

all:	picpac.so

clean:
	rm -rf *.o picpac.so

picpac.so:	lua-api.o picpac.o picpac-cv.o json11/json11.o
	g++ $(LDFLAGS) -o $@  $^ $(LDLIBS)

.cpp.o:
	g++ -c -o $@ $(CXXFLAGS) $(CXXFLAGS_EXTRA) $<

install:	picpac.so
	cp picpac.so $(LIBDIR)/lib

