/**
 * utils.cu: API definition for utility functions.
 * 
 * Copyright (c) Michele Collevati
 */


#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <limits.h>
#include <float.h>


#include "utils.cuh"
#include "launch_parameters_gpu.cuh"


/**
 * Tie breaking by selecting the smallest index.
 */
#define MIN_IDX


/**
 * Kernels
 */


/**
 * @brief Finds the index of the maximum int in a device array of positive
 * ints.
 * 
 * @param [in]num_thds_per_blk A number of threads per block.
 * @param [in]data A device array of positive ints.
 * @param [in]data_len Length of data.
 * @param [in]blk_vals A device array of the maximum values computed by each
 * block.
 * @param [in]blk_idxs A device array of the indices of the maximum values
 * computed by each block.
 * @param [in]blk_num A device counter to find the last block.
 * @param [out]idx_max_int The device index of the maximum int in data. -1 if
 * all ints in data are negative.
 * @retval None.
 */
__global__ void find_idx_max_int_krn(int num_thds_per_blk,
                                     int *data,
                                     int data_len,
                                     int *blk_vals,
                                     int *blk_idxs,
                                     int *blk_num,
                                     int *idx_max_int);


/**
 * @brief Finds the index of the maximum float in a device array of positive
 * floats.
 * 
 * @param [in]num_thds_per_blk A number of threads per block.
 * @param [in]data A device array of positive floats.
 * @param [in]data_len Length of data.
 * @param [in]blk_vals A device array of the maximum values computed by each
 * block.
 * @param [in]blk_idxs A device array of the indices of the maximum values
 * computed by each block.
 * @param [in]blk_num A device counter to find the last block.
 * @param [out]idx_max_float The device index of the maximum float in data. -1
 * if all floats in data are negative.
 * @retval None.
 */
__global__ void find_idx_max_float_krn(int num_thds_per_blk,
                                       float *data,
                                       int data_len,
                                       float *blk_vals,
                                       int *blk_idxs,
                                       int *blk_num,
                                       int *idx_max_float);


/**
 * @brief Finds the minimum int in a device array of ints.
 * 
 * @param [in]num_thds_per_blk A number of threads per block.
 * @param [in]data A device array of ints.
 * @param [in]data_len Length of data.
 * @param [in]blk_vals A device array of the minimum values computed by each
 * block.
 * @param [in]blk_num A device counter to find the last block.
 * @param [out]min_int The device minimum int in data.
 * @retval None.
 */
__global__ void find_min_int_krn(int num_thds_per_blk,
                                 int *data,
                                 int data_len,
                                 int *blk_vals,
                                 int *blk_num,
                                 int *min_int);


/**
 * @brief Finds the maximum float in a device array of floats.
 * 
 * @param [in]num_thds_per_blk A number of threads per block.
 * @param [in]data A device array of floats.
 * @param [in]data_len Length of data.
 * @param [in]blk_vals A device array of the maximum values computed by each
 * block.
 * @param [in]blk_num A device counter to find the last block.
 * @param [out]max_float The device maximum float in data.
 * @retval None.
 */
__global__ void find_max_float_krn(int num_thds_per_blk,
                                   float *data,
                                   int data_len,
                                   float *blk_vals,
                                   int *blk_num,
                                   float *max_float);


/**
 * @brief Initializes or sets int device memory to a int value.
 * Fills the first count ints of the memory area pointed to by ptr with the
 * constant int value value.
 *
 * @param [out]ptr Pointer to int device memory.
 * @param [in]value Value to set for each int of specified memory.
 * @param [in]count A number of ints to set.
 */
__global__ void cuda_memset_int_krn(int *ptr, int value, unsigned int count);


/**
 * @brief Initializes or sets float device memory to a float value.
 * Fills the first count floats of the memory area pointed to by ptr with the
 * constant float value value.
 *
 * @param [out]ptr Pointer to float device memory.
 * @param [in]value Value to set for each float of specified memory.
 * @param [in]count A number of floats to set.
 */
__global__ void cuda_memset_float_krn(float *ptr,
                                      float value,
                                      unsigned int count);


/**
 * API definition
 */


int find_idx_max_int(int *d_data, int data_len) {
    int num_blks = gpu_num_blocks(data_len);
    int num_thds_per_blk = gpu_num_threads_per_block();

    int *d_blk_vals;    /**
                         * Device array of the maximum values computed by each
                         * block.
                         */
    gpuErrchk( cudaMalloc((void**)&d_blk_vals,
                          sizeof *d_blk_vals * num_blks) );

    int *d_blk_idxs;    /**
                         * Device array of the indices of the maximum values
                         * computed by each block.
                         */
    gpuErrchk( cudaMalloc((void**)&d_blk_idxs,
                          sizeof *d_blk_idxs * num_blks) );

    int *d_blk_num;     /**
                         * Device counter to find the last block.
                         */
    gpuErrchk( cudaMalloc((void**)&d_blk_num,
                          sizeof *d_blk_num) );
    gpuErrchk( cudaMemset(d_blk_num, 0, sizeof *d_blk_num) );

    int *d_idx_max_int; /**
                         * Device index of the maximum int in d_data.
                         */
    gpuErrchk( cudaMalloc((void**)&d_idx_max_int,
                          sizeof *d_idx_max_int) );

    int vals_len = num_thds_per_blk;
    int idxs_len = num_thds_per_blk;
    int last_blk_len = 1;
    int shared_mem_size = (sizeof(int) * vals_len) +
                          (sizeof(int) * idxs_len) +
                          (sizeof(bool) * last_blk_len);

    find_idx_max_int_krn<<<num_blks, num_thds_per_blk, shared_mem_size>>>(
                                                            num_thds_per_blk,
                                                            d_data,
                                                            data_len,
                                                            d_blk_vals,
                                                            d_blk_idxs,
                                                            d_blk_num,
                                                            d_idx_max_int
                                                                         );

    gpuErrchk( cudaPeekAtLastError() );

    int idx_max_int;
    gpuErrchk( cudaMemcpy(&idx_max_int, d_idx_max_int,
                          sizeof idx_max_int,
                          cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaFree(d_blk_vals) );
    gpuErrchk( cudaFree(d_blk_idxs) );
    gpuErrchk( cudaFree(d_blk_num) );
    gpuErrchk( cudaFree(d_idx_max_int) );

    return idx_max_int;
}


int find_idx_max_float(float *d_data, int data_len) {
    int num_blks = gpu_num_blocks(data_len);
    int num_thds_per_blk = gpu_num_threads_per_block();

    float *d_blk_vals;      /**
                             * Device array of the maximum values computed by
                             * each block.
                             */
    gpuErrchk( cudaMalloc((void**)&d_blk_vals,
                          sizeof *d_blk_vals * num_blks) );

    int *d_blk_idxs;        /**
                             * Device array of the indices of the maximum
                             * values computed by each block.
                             */
    gpuErrchk( cudaMalloc((void**)&d_blk_idxs,
                          sizeof *d_blk_idxs * num_blks) );

    int *d_blk_num;         /**
                             * Device counter to find the last block.
                             */
    gpuErrchk( cudaMalloc((void**)&d_blk_num,
                          sizeof *d_blk_num) );
    gpuErrchk( cudaMemset(d_blk_num, 0, sizeof *d_blk_num) );

    int *d_idx_max_float;   /**
                             * Device index of the maximum float in d_data.
                             */
    gpuErrchk( cudaMalloc((void**)&d_idx_max_float,
                          sizeof *d_idx_max_float) );

    int vals_len = num_thds_per_blk;
    int idxs_len = num_thds_per_blk;
    int last_blk_len = 1;
    int shared_mem_size = (sizeof(float) * vals_len) +
                          (sizeof(int) * idxs_len) +
                          (sizeof(bool) * last_blk_len);

    find_idx_max_float_krn<<<num_blks, num_thds_per_blk, shared_mem_size>>>(
                                                            num_thds_per_blk,
                                                            d_data,
                                                            data_len,
                                                            d_blk_vals,
                                                            d_blk_idxs,
                                                            d_blk_num,
                                                            d_idx_max_float
                                                                         );

    gpuErrchk( cudaPeekAtLastError() );

    int idx_max_float;
    gpuErrchk( cudaMemcpy(&idx_max_float, d_idx_max_float,
                          sizeof idx_max_float,
                          cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaFree(d_blk_vals) );
    gpuErrchk( cudaFree(d_blk_idxs) );
    gpuErrchk( cudaFree(d_blk_num) );
    gpuErrchk( cudaFree(d_idx_max_float) );

    return idx_max_float;
}


int find_min_int(int *d_data, int data_len) {
    int num_blks = gpu_num_blocks(data_len);
    int num_thds_per_blk = gpu_num_threads_per_block();

    int *d_blk_vals;    /**
                         * Device array of the minimum values computed by each
                         * block.
                         */
    gpuErrchk( cudaMalloc((void**)&d_blk_vals,
                          sizeof *d_blk_vals * num_blks) );

    int *d_blk_num;     /**
                         * Device counter to find the last block.
                         */
    gpuErrchk( cudaMalloc((void**)&d_blk_num,
                          sizeof *d_blk_num) );
    gpuErrchk( cudaMemset(d_blk_num, 0, sizeof *d_blk_num) );

    int *d_min_int;     /**
                         * Device minimum int in d_data.
                         */
    gpuErrchk( cudaMalloc((void**)&d_min_int,
                          sizeof *d_min_int) );

    int vals_len = num_thds_per_blk;
    int last_blk_len = 1;
    int shared_mem_size = (sizeof(int) * vals_len) +
                          (sizeof(bool) * last_blk_len);

    find_min_int_krn<<<num_blks, num_thds_per_blk, shared_mem_size>>>(
                                                        num_thds_per_blk,
                                                        d_data,
                                                        data_len,
                                                        d_blk_vals,
                                                        d_blk_num,
                                                        d_min_int
                                                                     );

    gpuErrchk( cudaPeekAtLastError() );

    int min_int;
    gpuErrchk( cudaMemcpy(&min_int, d_min_int,
                          sizeof min_int,
                          cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaFree(d_blk_vals) );
    gpuErrchk( cudaFree(d_blk_num) );
    gpuErrchk( cudaFree(d_min_int) );

    return min_int;
}


float find_max_float(float *d_data, int data_len) {
    int num_blks = gpu_num_blocks(data_len);
    int num_thds_per_blk = gpu_num_threads_per_block();

    float *d_blk_vals;      /**
                             * Device array of the maximum values computed by
                             * each block.
                             */
    gpuErrchk( cudaMalloc((void**)&d_blk_vals,
                          sizeof *d_blk_vals * num_blks) );

    int *d_blk_num;         /**
                             * Device counter to find the last block.
                             */
    gpuErrchk( cudaMalloc((void**)&d_blk_num,
                          sizeof *d_blk_num) );
    gpuErrchk( cudaMemset(d_blk_num, 0, sizeof *d_blk_num) );

    float *d_max_float;       /**
                             * Device maximum float in d_data.
                             */
    gpuErrchk( cudaMalloc((void**)&d_max_float,
                          sizeof *d_max_float) );

    int vals_len = num_thds_per_blk;
    int last_blk_len = 1;
    int shared_mem_size = (sizeof(float) * vals_len) +
                          (sizeof(bool) * last_blk_len);

    find_max_float_krn<<<num_blks, num_thds_per_blk, shared_mem_size>>>(
                                                            num_thds_per_blk,
                                                            d_data,
                                                            data_len,
                                                            d_blk_vals,
                                                            d_blk_num,
                                                            d_max_float
                                                                       );

    gpuErrchk( cudaPeekAtLastError() );

    float max_float;
    gpuErrchk( cudaMemcpy(&max_float, d_max_float,
                          sizeof max_float,
                          cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaFree(d_blk_vals) );
    gpuErrchk( cudaFree(d_blk_num) );
    gpuErrchk( cudaFree(d_max_float) );

    return max_float;
}


__host__ cudaError_t cuda_memset_int(int *devPtr,
                                     int value,
                                     unsigned int count) {
    int num_blks = gpu_num_blocks(count);
    int num_thds_per_blk = gpu_num_threads_per_block();
    
    cuda_memset_int_krn<<<num_blks, num_thds_per_blk>>>(devPtr, value, count);

    cudaError_t code = cudaPeekAtLastError();

    return code;
}


__host__ cudaError_t cuda_memset_float(float *devPtr,
                                       float value,
                                       unsigned int count) {
    int num_blks = gpu_num_blocks(count);
    int num_thds_per_blk = gpu_num_threads_per_block();
    
    cuda_memset_float_krn<<<num_blks, num_thds_per_blk>>>(devPtr,
                                                          value,
                                                          count);

    cudaError_t code = cudaPeekAtLastError();

    return code;
}


/**
 * Kernel definitions
 */


__global__ void find_idx_max_int_krn(int num_thds_per_blk,
                                     int *data,
                                     int data_len,
                                     int *blk_vals,
                                     int *blk_idxs,
                                     int *blk_num,
                                     int *idx_max_int) {
    extern __shared__ volatile int array_fimik[];

    volatile int *vals = array_fimik;
    volatile int *idxs = vals + num_thds_per_blk;
    volatile bool *last_blk = (bool *)(idxs + num_thds_per_blk);

    int tid = threadIdx.x;
    int gid = tid + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    *last_blk = false;
    int val = -1;
    int idx = -1;

    // Sweep from global memory.
    while (gid < data_len) {
        if (data[gid] > val) {
            val = data[gid];
            idx = gid;
        }

        gid += stride;
    }

    // Populate shared memory.
    vals[tid] = val;
    idxs[tid] = idx;
    __syncthreads();

    // Sweep in shared memory.
    for (int s = (num_thds_per_blk >> 1); s > 0; s >>= 1) {
        if (tid < s) {
            if (vals[tid] < vals[tid + s]
#ifdef MIN_IDX
                || ((vals[tid] == vals[tid + s]) &&
                    (idxs[tid] > idxs[tid + s]))
#endif
               ) {
                vals[tid] = vals[tid + s];
                idxs[tid] = idxs[tid + s];
            }
        }

        __syncthreads();
    }

    // Perform block-level reduction.
    if (tid == 0) {
        blk_vals[blockIdx.x] = vals[0];
        blk_idxs[blockIdx.x] = idxs[0];

        if (atomicAdd(blk_num, 1) == gridDim.x - 1) {
            // True for the last block.
            *last_blk = true;
        }
    }

    __syncthreads();

    if (*last_blk) {
        val = -1;
        idx = -1;

        while (tid < gridDim.x) {
            if (blk_vals[tid] > val
#ifdef MIN_IDX
                || ((blk_vals[tid] == val) && (blk_idxs[tid] < idx))
#endif
               ) {
                val = blk_vals[tid];
                idx = blk_idxs[tid];
            }

            tid += blockDim.x;
        }

        tid = threadIdx.x;

        // Populate shared memory.
        vals[tid] = val;
        idxs[tid] = idx;
        __syncthreads();

        // Sweep in shared memory.
        for (int s = (num_thds_per_blk >> 1); s > 0; s >>= 1) {
            if (tid < s) {
                if (vals[tid] < vals[tid + s]
#ifdef MIN_IDX
                    || ((vals[tid] == vals[tid + s]) &&
                        (idxs[tid] > idxs[tid + s]))
#endif
                   ) {
                    vals[tid] = vals[tid + s];
                    idxs[tid] = idxs[tid + s];
                }
            }

            __syncthreads();
        }

        if (tid == 0) {
            *idx_max_int = idxs[0];
        }
    }
}


__global__ void find_idx_max_float_krn(int num_thds_per_blk,
                                       float *data,
                                       int data_len,
                                       float *blk_vals,
                                       int *blk_idxs,
                                       int *blk_num,
                                       int *idx_max_float) {
    extern __shared__ volatile float array_fimfk[];

    volatile float *vals = array_fimfk;
    volatile int *idxs = (int *)(vals + num_thds_per_blk);
    volatile bool *last_blk = (bool *)(idxs + num_thds_per_blk);

    int tid = threadIdx.x;
    int gid = tid + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    *last_blk = false;
    float val = -1.0;
    int idx = -1;

    // Sweep from global memory.
    while (gid < data_len) {
        if (data[gid] > val) {
            val = data[gid];
            idx = gid;
        }

        gid += stride;
    }

    // Populate shared memory.
    vals[tid] = val;
    idxs[tid] = idx;
    __syncthreads();

    // Sweep in shared memory.
    for (int s = (num_thds_per_blk >> 1); s > 0; s >>= 1) {
        if (tid < s) {
            if (vals[tid] < vals[tid + s]
#ifdef MIN_IDX
                || ((vals[tid] == vals[tid + s]) &&
                    (idxs[tid] > idxs[tid + s]))
#endif
               ) {
                vals[tid] = vals[tid + s];
                idxs[tid] = idxs[tid + s];
            }
        }

        __syncthreads();
    }

    // Perform block-level reduction.
    if (tid == 0) {
        blk_vals[blockIdx.x] = vals[0];
        blk_idxs[blockIdx.x] = idxs[0];

        if (atomicAdd(blk_num, 1) == gridDim.x - 1) {
            // True for the last block.
            *last_blk = true;
        }
    }

    __syncthreads();

    if (*last_blk) {
        val = -1.0;
        idx = -1;

        while (tid < gridDim.x) {
            if (blk_vals[tid] > val
#ifdef MIN_IDX
                || ((blk_vals[tid] == val) && (blk_idxs[tid] < idx))
#endif
               ) {
                val = blk_vals[tid];
                idx = blk_idxs[tid];
            }

            tid += blockDim.x;
        }

        tid = threadIdx.x;

        // Populate shared memory.
        vals[tid] = val;
        idxs[tid] = idx;
        __syncthreads();

        // Sweep in shared memory.
        for (int s = (num_thds_per_blk >> 1); s > 0; s >>= 1) {
            if (tid < s) {
                if (vals[tid] < vals[tid + s]
#ifdef MIN_IDX
                    || ((vals[tid] == vals[tid + s]) &&
                        (idxs[tid] > idxs[tid + s]))
#endif
                   ) {
                    vals[tid] = vals[tid + s];
                    idxs[tid] = idxs[tid + s];
                }
            }

            __syncthreads();
        }

        if (tid == 0) {
            *idx_max_float = idxs[0];
        }
    }
}


__global__ void find_min_int_krn(int num_thds_per_blk,
                                 int *data,
                                 int data_len,
                                 int *blk_vals,
                                 int *blk_num,
                                 int *min_int) {
    extern __shared__ volatile int array_fmik[];

    volatile int *vals = array_fmik;
    volatile bool *last_blk = (bool *)(vals + num_thds_per_blk);

    int tid = threadIdx.x;
    int gid = tid + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    *last_blk = false;
    int val = INT_MAX;

    // Sweep from global memory.
    while (gid < data_len) {
        if (data[gid] < val) {
            val = data[gid];
        }

        gid += stride;
    }

    // Populate shared memory.
    vals[tid] = val;
    __syncthreads();

    // Sweep in shared memory.
    for (int s = (num_thds_per_blk >> 1); s > 0; s >>= 1) {
        if (tid < s) {
            if (vals[tid] > vals[tid + s]) {
                vals[tid] = vals[tid + s];
            }
        }

        __syncthreads();
    }

    // Perform block-level reduction.
    if (tid == 0) {
        blk_vals[blockIdx.x] = vals[0];
        
        if (atomicAdd(blk_num, 1) == gridDim.x - 1) {
            // True for the last block.
            *last_blk = true;
        }
    }

    __syncthreads();

    if (*last_blk) {
        val = INT_MAX;

        while (tid < gridDim.x) {
            if (blk_vals[tid] < val) {
                val = blk_vals[tid];
            }

            tid += blockDim.x;
        }

        tid = threadIdx.x;

        // Populate shared memory.
        vals[tid] = val;
        __syncthreads();

        // Sweep in shared memory.
        for (int s = (num_thds_per_blk >> 1); s > 0; s >>= 1) {
            if (tid < s) {
                if (vals[tid] > vals[tid + s]) {
                    vals[tid] = vals[tid + s];
                }
            }

            __syncthreads();
        }

        if (tid == 0) {
            *min_int = vals[0];
        }
    }
}


__global__ void find_max_float_krn(int num_thds_per_blk,
                                   float *data,
                                   int data_len,
                                   float *blk_vals,
                                   int *blk_num,
                                   float *max_float) {
    extern __shared__ volatile float array_fmfk[];

    volatile float *vals = array_fmfk;
    volatile bool *last_blk = (bool *)(vals + num_thds_per_blk);

    int tid = threadIdx.x;
    int gid = tid + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    *last_blk = false;
    float val = -FLT_MAX;

    // Sweep from global memory.
    while (gid < data_len) {
        if (data[gid] > val) {
            val = data[gid];
        }

        gid += stride;
    }

    // Populate shared memory.
    vals[tid] = val;
    __syncthreads();

    // Sweep in shared memory.
    for (int s = (num_thds_per_blk >> 1); s > 0; s >>= 1) {
        if (tid < s) {
            if (vals[tid] < vals[tid + s]) {
                vals[tid] = vals[tid + s];
            }
        }

        __syncthreads();
    }

    // Perform block-level reduction.
    if (tid == 0) {
        blk_vals[blockIdx.x] = vals[0];

        if (atomicAdd(blk_num, 1) == gridDim.x - 1) {
            // True for the last block.
            *last_blk = true;
        }
    }

    __syncthreads();

    if (*last_blk) {
        val = -FLT_MAX;

        while (tid < gridDim.x) {
            if (blk_vals[tid] > val) {
                val = blk_vals[tid];
            }

            tid += blockDim.x;
        }

        tid = threadIdx.x;

        // Populate shared memory.
        vals[tid] = val;
        __syncthreads();

        // Sweep in shared memory.
        for (int s = (num_thds_per_blk >> 1); s > 0; s >>= 1) {
            if (tid < s) {
                if (vals[tid] < vals[tid + s]) {
                    vals[tid] = vals[tid + s];
                }
            }

            __syncthreads();
        }

        if (tid == 0) {
            *max_float = vals[0];
        }
    }
}


__global__ void cuda_memset_int_krn(int *ptr, int value, unsigned int count) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (gid < count) {
        ptr[gid] = value;

        gid += stride;
    }
}


__global__ void cuda_memset_float_krn(float *ptr,
                                      float value,
                                      unsigned int count) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (gid < count) {
        ptr[gid] = value;

        gid += stride;
    }
}
