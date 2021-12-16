/**
 * launch_parameters_gpu.cuh: API definition for setting up CUDA.
 * 
 * Copyright (c) Michele Collevati
 */


#ifndef LAUNCH_PARAMETERS_GPU_CUH
#define LAUNCH_PARAMETERS_GPU_CUH


#include "utils.cuh"


/**
 * Global variables
 */


static int gpu_num_thds_per_blk = 512;          // Number of threads per block.
static int gpu_multiproc_num_blks;              /**
                                                 * Multiprocessors-based number
                                                 * of blocks per kernel launch.
                                                 */
static bool gpu_set_multiproc_num_blks = false; /**
                                                 * Flag to check if
                                                 * gpu_multiproc_num_blks has
                                                 * been set.
                                                 */
static bool gpu_set_dev = false;                /**
                                                 * Flag to check if the device
                                                 * has been set.
                                                 */


/**
 * Auxiliary functions
 */


/**
 * @brief Sets the number of blocks per kernel launch based on the number of
 * multiprocessors on the set device.
 * 
 * @retval None.
 */
inline void gpu_set_multiproc_num_blocks();


/**
 * API
 */


/**
 * @brief Sets the device.
 * 
 * @param [in]device A device on which the active host thread should execute
 * the device code.
 * @retval None.
 */
inline void gpu_set_device(int device) {
    gpuErrchk( cudaSetDevice(device) );
    gpu_set_dev = true;
    gpu_set_multiproc_num_blks = false;
}


/**
 * @brief Sets the number of threads per block.
 * 
 * @param [in]num_threads_per_block A number of threads per block.
 * @retval None.
 */
inline void gpu_set_num_threads_per_block(int num_threads_per_block) {
    gpu_num_thds_per_blk = num_threads_per_block;
}


/**
 * @brief Gets the number of blocks per kernel launch based on the number of
 * multiprocessors on the set device.
 * 
 * @retval The number of blocks.
 */
inline int gpu_num_blocks() {
    if (!gpu_set_multiproc_num_blks) {
        gpu_set_multiproc_num_blocks();
    }

    return gpu_multiproc_num_blks;
}


/**
 * @brief Gets the number of blocks per kernel launch based on the length of
 * data to be processed.
 * 
 * @param [in]data_len Length of data to be processed.
 * @retval The number of blocks.
 */
inline int gpu_num_blocks(int data_len) {
    if (!gpu_set_multiproc_num_blks) {
        gpu_set_multiproc_num_blocks();
    }

    // Maximum number of blocks needed.
    int max_num_blks_needed = (data_len + gpu_num_thds_per_blk - 1) /
                              gpu_num_thds_per_blk;

    return min(max_num_blks_needed, gpu_multiproc_num_blks);
}


/**
 * @brief Gets the number of threads per block.
 * 
 * @retval The number of threads per block.
 */
inline int gpu_num_threads_per_block() {
    return gpu_num_thds_per_blk;
}


/**
 * Auxiliary function definitions
 */


inline void gpu_set_multiproc_num_blocks() {
    if (!gpu_set_dev) {
        gpu_set_device(0);
    }

    int device;
    gpuErrchk( cudaGetDevice(&device) );
    cudaDeviceProp dev_prop;
    gpuErrchk( cudaGetDeviceProperties(&dev_prop, device) );
    gpu_multiproc_num_blks = dev_prop.multiProcessorCount * 2;
    gpu_set_multiproc_num_blks = true;
}


#endif
