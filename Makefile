# Tunel de Vento Virtual 3D - LBM D3Q19
# Compila as tres versoes: Serial, OpenMP e CUDA.
#
# Uso:
#   make            -> compila serial, openmp e cuda (cuda falha silenciosamente
#                       se o nvcc nao estiver instalado nesta maquina)
#   make serial
#   make openmp
#   make cuda
#   make viewport   -> janela interativa (precisa do raylib em third_party/)
#   make RES="-DNX=64 -DNY=32 -DNZ=32"   -> muda a resolucao da grade
#   make clean

CC      ?= gcc
NVCC    ?= nvcc
CFLAGS  := -O3 -std=c11 -Wall
OMPFLAGS:= -fopenmp
NVFLAGS := -O3 -std=c++11

RAYLIB_DIR := third_party/raylib-6.2_win64_mingw-w64
RAYLIB_LIBS:= -lraylib -lopengl32 -lgdi32 -lwinmm

RES ?=

all: serial openmp cuda

serial: src/serial/lbm_serial.c src/common/lbm_params.h src/common/lbm_io.h
	$(CC) $(CFLAGS) $(RES) -o lbm_serial src/serial/lbm_serial.c -lm

openmp: src/openmp/lbm_openmp.c src/common/lbm_params.h src/common/lbm_io.h
	$(CC) $(CFLAGS) $(OMPFLAGS) $(RES) -o lbm_openmp src/openmp/lbm_openmp.c -lm

cuda: src/cuda/lbm_cuda.cu src/common/lbm_params.h src/common/lbm_io.h
	$(NVCC) $(NVFLAGS) $(RES) -o lbm_cuda src/cuda/lbm_cuda.cu

viewport: src/viewport/lbm_viewport.c src/common/lbm_params.h
	$(CC) $(CFLAGS) $(OMPFLAGS) $(RES) -I$(RAYLIB_DIR)/include -o lbm_viewport src/viewport/lbm_viewport.c -L$(RAYLIB_DIR)/lib $(RAYLIB_LIBS) -lm

clean:
	rm -f lbm_serial lbm_openmp lbm_cuda lbm_viewport *.pgm

.PHONY: all serial openmp cuda viewport clean
