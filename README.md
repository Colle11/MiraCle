# README

1. Specify the path to the `nvcc` compiler in the `NVCC` variable of the *Makefile*.
2. Specify the CUDA compute capability of the GPU in the `NVCCFLAGS` variable of the *Makefile*.
3. Compile using: `make`.
4. Use as follows: `./build/bin/miracle filename`, where `filename` is a SAT formula in DIMACS CNF format.
5. Clean the directory using: `make clean`.

