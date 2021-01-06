# Automatic variables:
# $@: the file name of the target of the rule.
# $<: the name of the first prerequisite.
# $^: the names of all the prerequisites, with spaces between them.

.PHONY = all clean

NVCC = nvcc
CC = gcc

# CUDA_INCLUDEPATH = /opt/cuda/include
# CUDA_LIBPATH = /opt/cuda/lib

NVCC_OPTS = -O3 -arch=sm_50 -Xcompiler -Wall -Xcompiler -Wextra -m64
CC_OPTS = -O3 -Wall -Wextra -m64
# LD_OPTS = -lm

BIN = miracle
OBJS = driver.o miracle.o formula.o

all: $(BIN)

$(BIN): $(OBJS)
		$(NVCC) $(NVCC_OPTS) $^ -o $@

driver.o: driver.cu miracle.h
		$(NVCC) -c $(NVCC_OPTS) $<

miracle.o: miracle.cu miracle.h formula.h
		$(NVCC) -c $(NVCC_OPTS) $<

formula.o: formula.cu formula.h utilities.h
		$(NVCC) -c $(NVCC_OPTS) $<

clean:
		rm -f $(OBJS) $(BIN)
