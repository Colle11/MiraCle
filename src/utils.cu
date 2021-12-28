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
// #include "miracle.cuh"
// #include "miracle_dynamic.cuh"


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


// void print_mrc_vs_mrcdyn_debug_info(Miracle *mrc, Miracle_Dyn *mrc_dyn) {
//     printf("****************************************************************");
//     printf("\n");
//     printf("*********************    DEBUG INFO    *************************");
//     printf("\n");
//     printf("****************************************************************");
//     printf("\n\n");

//     mrc_print_miracle(mrc);

//     printf("****************************************************************");
//     printf("\n\n");

//     mrc_dyn_print_miracle(mrc_dyn);

//     printf("****************************************************************");
//     printf("\n\n");

//     printf("*** Correctness test ***\n\n");

//     printf("*** Variable assignments test ***\n\n");

//     for (Var v = 0; v < mrc->var_ass_len; v++) {
//         if (mrc->var_ass[v] != mrc_dyn->var_ass[v]) {
//             printf("mrc->var_ass[%d] = %d", v, mrc->var_ass[v]);
//             printf("    !=    ");
//             printf("mrc_dyn->var_ass[%d] = %d\n", v, mrc_dyn->var_ass[v]);
//             exit(EXIT_FAILURE);
//         }
//     }

//     printf("OK!\n");

//     printf("\n*** End variable assignments test ***\n\n");

//     printf("*** Clause satisfiability test ***\n\n");

//     for (int c = 0; c < mrc->clause_sat_len; c++) {
//         if (mrc->clause_sat[c] != mrc_dyn->clause_sat[c]) {
//             printf("mrc->clause_sat[%d] = %d", c, mrc->clause_sat[c]);
//             printf("    !=    ");
//             printf("mrc_dyn->clause_sat[%d] = %d\n", c, mrc_dyn->clause_sat[c]);
//             exit(EXIT_FAILURE);
//         }
//     }

//     printf("OK!\n");

//     printf("\n*** End clause satisfiability test ***\n\n");

//     printf("*** Unresolved clause current sizes test ***\n\n");

//     int c_size;     // Clause size.
//     Lidx lidx;
//     Var var;
//     // Array of clause current sizes derived from mrc.
//     int *clause_sizes = (int *)malloc(sizeof *clause_sizes *
//                                       mrc->phi->num_clauses);

//     for (int c = 0; c < mrc->clause_sat_len; c++) {
//         c_size = 0;

//         if (!(mrc->clause_sat[c])) {
//             for (int l = mrc->phi->clause_indices[c];
//                  l < mrc->phi->clause_indices[c+1];
//                  l++) {
//                 lidx = mrc->phi->clauses[l];
//                 var = lidx_to_var(lidx);

//                 if (!(mrc->var_ass[var])) {
//                     c_size++;
//                 }
//             }
//         }

//         clause_sizes[c] = c_size;
//     }

//     for (int c = 0; c < mrc_dyn->clause_size_len; c++) {
//         if (!mrc->clause_sat[c] &&
//             (clause_sizes[c] != mrc_dyn->unres_clause_size[c])) {
//                 printf("clause_sizes[%d] = %d", c, clause_sizes[c]);
//                 printf("    !=    ");
//                 printf("mrc_dyn->unres_clause_size[%d] = %d\n",
//                        c, mrc_dyn->unres_clause_size[c]);
//                 exit(EXIT_FAILURE);
//         }
//     }

//     printf("OK!\n");

//     printf("\n*** End unresolved clause current sizes test ***\n\n");

//     printf("*** Unresolved literal current occurrences test ***\n\n");

//     // Array of literal current occurrences derived from mrc.
//     int *lit_occ = (int *)calloc(mrc->phi->num_vars * 2, sizeof *lit_occ);

//     for (int c = 0; c < mrc->clause_sat_len; c++) {
//         if (!(mrc->clause_sat[c])) {
//             for (int l = mrc->phi->clause_indices[c];
//                  l < mrc->phi->clause_indices[c+1];
//                  l++) {
//                 lidx = mrc->phi->clauses[l];
//                 var = lidx_to_var(lidx);

//                 if (!(mrc->var_ass[var])) {
//                     lit_occ[lidx]++;
//                 }
//             }
//         }
//     }

//     for (int l = 0; l < mrc_dyn->lit_occ_len; l++) {
//         if (!mrc->var_ass[lidx_to_var(l)] &&
//             (lit_occ[l] != mrc_dyn->unres_lit_occ[l])) {
//                 printf("lit_occ[%d] = %d", l, lit_occ[l]);
//                 printf("    !=    ");
//                 printf("mrc_dyn->unres_lit_occ[%d] = %d\n",
//                        l, mrc_dyn->unres_lit_occ[l]);
//                 exit(EXIT_FAILURE);
//         }
//     }

//     printf("OK!\n");

//     printf("\n*** End unresolved literal current occurrences test ***\n\n");

//     printf("*** End correctness test ***\n\n");

//     printf("****************************************************************");
//     printf("\n");
//     printf("********************    END DEBUG INFO    **********************");
//     printf("\n");
//     printf("****************************************************************");
//     printf("\n\n");
// }


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
