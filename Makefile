LIBS_PATH=/home/andrew/PhD/code/measurement/src/c/
PCM_PATH=/home/andrew/Downloads/IntelPerformanceCounterMonitorV2.8
CORE=0

all: dist/main-nopcm dist/main-pcm

dist/main-nopcm: src/main-nopcm.cpp
	@ mkdir -p dist
	@ g++ --std=c++11 -O3 -msse4.2 -I$(LIBS_PATH) -I ./src/ src/main-nopcm.cpp -o dist/main-nopcm

dist/main-pcm: src/main-pcm.cpp src/test.hpp
	@ mkdir -p dist
	@ ln -s -f $(PCM_PATH) pcm
	@ make -C pcm
	@ g++ --std=c++11 -O3 -msse4.2 \
	-I$(LIBS_PATH) -I ./pcm/  -I ./src \
	-L ./pcm/cpucounters.o -L ./pcm/msr.o -L ./pcm/pci.o -L ./pcm/client_bw.o -lpthread \
	./pcm/cpucounters.cpp ./pcm/msr.cpp ./pcm/pci.cpp ./pcm/client_bw.cpp src/main-pcm.cpp -o dist/main-pcm

clean:
	@ rm -rf dist/ pcm

run-pcm: dist/main-pcm
	sudo numactl --physcpubind=$(CORE) nice -20 ./dist/main-pcm
