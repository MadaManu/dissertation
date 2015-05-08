LIBS_PATH=/home/andrew/PhD/code/measurement/src/c/
PCM_PATH=/home/andrew/Downloads/IntelPerformanceCounterMonitorV2.8
CORE=0
SIZE=256
STATS_EXE=/home/andrew/PhD/code/measurement/dist/statistics
HIST_EXE=/home/andrew/PhD/code/measurement/dist/histogram
KDE_SH=/home/andrew/PhD/code/measurement/src/sh/kde.sh

all: dist/main-nopcm dist/main-pcm

dist/main-nopcm: src/main-nopcm.cpp
	@ mkdir -p dist
	@ g++ --std=c++11 -O3 -msse4.2 -I$(LIBS_PATH) -DSIZE=$(SIZE) -I ./src/ src/main-nopcm.cpp -o dist/main-nopcm

dist/main-pcm: src/main-pcm.cpp src/test.hpp
	@ mkdir -p dist
	@ ln -s -f $(PCM_PATH) pcm
	@ make -C pcm
	@ g++ --std=c++11 -O3 -msse4.2 -DSIZE=$(SIZE) \
	-I$(LIBS_PATH) -I ./pcm/  -I ./src \
	-L ./pcm/cpucounters.o -L ./pcm/msr.o -L ./pcm/pci.o -L ./pcm/client_bw.o -lpthread \
	./pcm/cpucounters.cpp ./pcm/msr.cpp ./pcm/pci.cpp ./pcm/client_bw.cpp src/main-pcm.cpp -o dist/main-pcm

clean:
	@ rm -rf dist/ pcm

stats-clean:
	@ rm -rf results/*/*/*.dat results/*/*/*.stats results/*/*/*.pdf

run-pcm: dist/main-pcm
	sudo numactl --physcpubind=$(CORE) nice -20 ./dist/main-pcm
	sudo chown -R $(USER):users results

draw-pcm-1: results/level1/$(SIZE)
	cd results/level1/$(SIZE)/ ; \
	for x in `ls *cycles.txt`; do $(STATS_EXE) $$x 100 20 0 1100 > $$x.stats; done ; \
	for x in `ls *cycles.txt`; do $(HIST_EXE) $$x 100 20 0 1100 > $$x.dat; done ; \
	OUTPUT=level1.pdf $(KDE_SH) *.dat

draw-pcm-2: results/level2/$(SIZE)
	cd results/level2/$(SIZE)/ ; \
	for x in `ls *cycles.txt`; do $(STATS_EXE) $$x 100 20 0 1000000 > $$x.stats; done ; \
	for x in `ls *cycles.txt`; do $(HIST_EXE) $$x 100 20 0 1000000 > $$x.dat; done ; \
	OUTPUT=level2.pdf $(KDE_SH) *.dat

draw-pcm-3: results/level3/$(SIZE)
	cd results/level3/$(SIZE)/ ; \
	for x in `ls *cycles.txt`; do $(STATS_EXE) $$x 100 20 0 1000000000 > $$x.stats; done ; \
	for x in `ls *cycles.txt`; do $(HIST_EXE) $$x 100 20 0 1000000000 > $$x.dat; done ; \
	OUTPUT=level3.pdf $(KDE_SH) *.dat
