# Load CUDA using the following command
# module load cuda
#
CC = nvcc
# CFLAGS = -arch compute_20
# NVCCFLAGS = -arch compute_20
CFLAGS = -O3 -arch=compute_37 -code=sm_37  -D_FORCE_INLINES
NVCCFLAGS = -O3 -arch=compute_37 -code=sm_37  -D_FORCE_INLINES
LIBS = 

TARGETS = serial gpu autograder

all:	$(TARGETS)

serial: serial.o common.o
	$(CC) -o $@ $(LIBS) serial.o common.o
gpu: gpu.o common.o
	$(CC) -o $@ $(NVCCLIBS) gpu.o common.o
autograder: autograder.o common.o
	$(CC) -o $@ $(LIBS) autograder.o common.o

serial.o: serial.cu common.h
	$(CC) -c $(CFLAGS) serial.cu
autograder.o: autograder.cu common.h
	$(CC) -c $(CFLAGS) autograder.cu
gpu.o: gpu.cu common.h
	$(CC) -c $(NVCCFLAGS) gpu.cu
common.o: common.cu common.h
	$(CC) -c $(CFLAGS) common.cu

clean:
	rm -f *.o $(TARGETS) *.stdout *.txt
