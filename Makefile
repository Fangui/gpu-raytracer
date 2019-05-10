NVCC ?= nvcc
VPATH=src/
CXXFLAGS += -Wall -Wextra -std=c++17 -pedantic -O3 -flto -fopenmp -march=native # for perf -g3 -fno-omit-frame-pointer
CUDA_FLAG = -Xcompiler -fopenmp
CXXLIBS += -lSDL2 -lSDL2_image

SRC = kdtree.cc triangle.cc material.cc parse.cc light.cc \
	compute_light.cc matrix.cc texture.cc sphere_light.cc

CUDA_SRC = src/vector.cu src/main.cu
OBJ = ${SRC:.cc=.o}
BIN = main

all: $(BIN)

main:  ${OBJ}
	$(NVCC) -rdc=true $(CUDA_FLAG) $(CUDA_SRC) $(OBJ) $(CXXLIBS) -o $(BIN)

check: CXXFLAGS += -g3 -O0 -fno-inline -fsanitize=address
check: $(BIN)

.PHONY: clean check
clean:
	${RM} ${OBJ}
	${RM} $(BIN)

tar:
	tar -cvjf examples.tar.bz2 examples Textures
untar:
	tar -xvf examples.tar.bz2
