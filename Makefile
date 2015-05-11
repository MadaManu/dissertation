LIBS_PATH=/home/andrew/workspace/measurement/src/c/
PCM_PATH=/home/andrew/Downloads/IntelPerformanceCounterMonitorV2.8
CORE=0
STATS_EXE=/home/andrew/workspace/measurement/dist/statistics
HIST_EXE=/home/andrew/workspace/measurement/dist/histogram
KDE_SH=/home/andrew/workspace/measurement/src/sh/kde.sh

all: dist/main-pcm

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
	sudo modprobe msr
	sudo numactl --physcpubind=$(CORE) nice -20 ./dist/main-pcm
	sudo chown -R $(USER):users results

draw-pcm-1: results/level1/$(SIZE)
	cd results/level1/$(SIZE)/ ; \
	for x in `ls *cycles.txt`; do $(STATS_EXE) $$x 100 20 0 1100 > $$x.stats; done ; \
	for x in `ls *cycles.txt`; do $(HIST_EXE) $$x 100 20 0 1100 > $$x.dat; done ; \
	OUTPUT=level1-cycles.pdf $(KDE_SH) *cycles.txt.dat

draw-ipc-1: results/level1/$(SIZE)
	cd results/level1/$(SIZE)/ ; \
	for x in `ls *ipc.txt`; do $(STATS_EXE)-float $$x 100 0.01 0 10 > $$x.stats; done ; \
	for x in `ls *ipc.txt`; do $(HIST_EXE)-float $$x 100 0.01 0 10 > $$x.dat; done ; \
	LINESTYLE=points XLABEL="Instructions per clock" OUTPUT=level1-ipc.pdf $(KDE_SH) *ipc.txt.dat

draw-l2h-1: results/level1/$(SIZE)
	cd results/level1/$(SIZE)/ ; \
	for x in `ls *l2h.txt`; do $(STATS_EXE)-float $$x 100 0.01 0 10 > $$x.stats; done ; \
	for x in `ls *l2h.txt`; do $(HIST_EXE)-float $$x 100 0.01 0 10 > $$x.dat; done ; \
	LINESTYLE=points XLABEL="L2 Hit Rate" OUTPUT=level1-l2h.pdf $(KDE_SH) *l2h.txt.dat

draw-l3h-1: results/level1/$(SIZE)
	cd results/level1/$(SIZE)/ ; \
	for x in `ls *l3h.txt`; do $(STATS_EXE)-float $$x 100 0.01 0 10 > $$x.stats; done ; \
	for x in `ls *l3h.txt`; do $(HIST_EXE)-float $$x 100 0.01 0 10 > $$x.dat; done ; \
	LINESTYLE=points XLABEL="L3 Hit Rate" OUTPUT=level1-l3h.pdf $(KDE_SH) *l3h.txt.dat

draw-pcm-2: results/level2/$(SIZE)
	cd results/level2/$(SIZE)/ ; \
	for x in `ls *cycles.txt`; do $(STATS_EXE) $$x 100 6000 0 1000000 > $$x.stats; done ; \
	for x in `ls *cycles.txt`; do $(HIST_EXE) $$x 100 6000 0 1000000 > $$x.dat; done ; \
	OUTPUT=level2.pdf $(KDE_SH) *cycles.txt.dat

draw-pcm-3: results/level3/$(SIZE)
	cd results/level3/$(SIZE)/ ; \
	for x in `ls *cycles.txt`; do $(STATS_EXE) $$x 100 300000 0 1000000000 > $$x.stats; done ; \
	for x in `ls *cycles.txt`; do $(HIST_EXE) $$x 100 300000 0 1000000000 > $$x.dat; done ; \
	OUTPUT=level3.pdf $(KDE_SH) *cycles.txt.dat

draw-pcm-3-sse: results/level3/$(SIZE)
	cd results/level3/$(SIZE)/ ; \
	for x in `ls *SSE*cycles.txt`; do $(STATS_EXE) $$x 100 100 0 3000 > $$x.stats; done ; \
	for x in `ls *SSE*cycles.txt`; do $(HIST_EXE) $$x 100 100 0 3000 > $$x.dat; done ; \
	OUTPUT=level3-SSE.pdf $(KDE_SH) *SSE*cycles.txt.dat
