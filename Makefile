LIBS_PATH=/home/andrew/PhD/code/measurement/src/c/

all: dist/main

dist/main: main.cpp
	@ mkdir -p dist
	@ g++ --std=c++11 -O3 -msse4.2 -I$(LIBS_PATH) main.cpp -o dist/main

clean:
	@ rm -rf dist/
