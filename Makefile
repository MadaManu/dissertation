LIBS_PATH=/home/andrew/PhD/code/measurement/src/c/
PCM_PATH=/home/andrew/Tools/IntelPerformanceCounterMonitorV2.8
CORE=0
STATS_EXE=/home/andrew/PhD/code/measurement/dist/statistics
HIST_EXE=/home/andrew/PhD/code/measurement/dist/histogram
KDE_SH=/home/andrew/PhD/code/measurement/src/sh/kde.sh

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

dist/main-pcm-branchless: src/main-pcm.cpp src/test.hpp
	@ mkdir -p dist
	@ ln -s -f $(PCM_PATH) pcm
	@ make -C pcm
	@ g++ --std=c++11 -O3 -msse4.2 -DSIZE=$(SIZE) -DUSE_GENERATED \
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

# draw-pcm-1: results/level1/$(SIZE)
# 	cd results/level1/$(SIZE)/ ; \
# 	for x in `ls *cycles.txt`; do $(STATS_EXE) $$x 100 20 0 1100 > $$x.stats; done ; \
# 	for x in `ls *cycles.txt`; do $(HIST_EXE) $$x 100 20 0 1100 > $$x.dat; done ; \
# 	OUTPUT=level1.pdf $(KDE_SH) *.dat

draw-dot-prod: results/level3/$(SIZE)
	cd results/level3/$(SIZE)/ ; \
	for x in `ls *cycles.txt`; do $(STATS_EXE) $$x 100 100 0 8888519900000 > $$x.stats; done ; \
	for x in `ls *cycles.txt`; do $(HIST_EXE) $$x 100 100 0 8888519900000 > $$x.dat; done ; \
	OUTPUT=mat-mat-sse-$(SIZE).pdf $(KDE_SH) matrix-matrix-mul-SSE-double-cycles.txt.dat matrix-matrix-mul-SSE-f48-cycles.txt.dat

draw-pcm-2: results/level2/$(SIZE)
	cd results/level2/$(SIZE)/ ; \
	for x in `ls *cycles.txt`; do $(STATS_EXE) $$x 100 1000 0 1000000 > $$x.stats; done ; \
	for x in `ls *cycles.txt`; do $(HIST_EXE) $$x 100 1000 0 1000000 1 > $$x.dat; done ; \
	OUTPUT=level2.pdf $(KDE_SH) *double*cycles.txt.dat

draw-ipc-2: results/level2/$(SIZE)
	cd results/level2/$(SIZE)/ ; \
	for x in `ls *ipc.txt`; do $(STATS_EXE)-float $$x 100 0.2 0 10 > $$x.stats; done ; \
	for x in `ls *ipc.txt`; do $(HIST_EXE)-float $$x 100 0.2 0 10 > $$x.dat; done ; \
	XLABEL="Instructions per clock" OUTPUT=level2-ipc.pdf $(KDE_SH) *double*ipc.txt.dat

draw-l2h-2: results/level2/$(SIZE)
	cd results/level2/$(SIZE)/ ; \
	for x in `ls *l2h.txt`; do $(STATS_EXE)-float $$x 100 0.2 0 10 > $$x.stats; done ; \
	for x in `ls *l2h.txt`; do $(HIST_EXE)-float $$x 100 0.2 0 10 > $$x.dat; done ; \
	XLABEL="L2 Hit Rate" OUTPUT=level2-l2h.pdf $(KDE_SH) *double*l2h.txt.dat

draw-l3h-2: results/level2/$(SIZE)
	cd results/level2/$(SIZE)/ ; \
	for x in `ls *l3h.txt`; do $(STATS_EXE)-float $$x 100 0.2 0 10 > $$x.stats; done ; \
	for x in `ls *l3h.txt`; do $(HIST_EXE)-float $$x 100 0.2 0 10 > $$x.dat; done ; \
	XLABEL="L3 Hit Rate" OUTPUT=level2-l3h.pdf $(KDE_SH) *double*l3h.txt.dat

draw-pcm-3: results/level3/$(SIZE)
	cd results/level3/$(SIZE)/ ; \
	for x in `ls *cycles.txt`; do $(STATS_EXE) $$x 100 300000 0 1000000000 > $$x.stats; done ; \
	for x in `ls *cycles.txt`; do $(HIST_EXE) $$x 100 300000 0 1000000000 1 > $$x.dat; done ; \
	OUTPUT=level3.pdf $(KDE_SH) *cycles.txt.dat

draw-pcm-3-sse: results/level3/$(SIZE)
	cd results/level3/$(SIZE)/ ; \
	for x in `ls *SSE*cycles.txt`; do $(STATS_EXE) $$x 100 100 0 3000 > $$x.stats; done ; \
	for x in `ls *SSE*cycles.txt`; do $(HIST_EXE) $$x 100 100 0 3000 1 > $$x.dat; done ; \
	OUTPUT=level3-SSE.pdf $(KDE_SH) *SSE*cycles.txt.dat
