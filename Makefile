CC = gcc
NVCC = nvcc

CFLAGS = -O3 -Wall -Iinclude
NVCCFLAGS = -O3 -Iinclude -Xcompiler -fPIC

TARGET = kudann_bench

all: $(TARGET)

src/knn_cpu.o: src/knn_cpu.c include/kudann.h
	$(CC) $(CFLAGS) -c src/knn_cpu.c -o src/knn_cpu.o

src/classifier.o: src/classifier.c include/kudann.h
	$(CC) $(CFLAGS) -c src/classifier.c -o src/classifier.o

src/knn_cuda.o: src/knn_cuda.cu include/kudann.h
	$(NVCC) $(NVCCFLAGS) -c src/knn_cuda.cu -o src/knn_cuda.o

$(TARGET): src/knn_cpu.o src/classifier.o src/knn_cuda.o main.c
	$(NVCC) $(NVCCFLAGS) src/knn_cpu.o src/classifier.o src/knn_cuda.o main.c -lm -o $(TARGET)

clean:
	rm -f src/*.o $(TARGET)