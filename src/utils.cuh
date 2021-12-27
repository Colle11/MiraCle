/**
 * utils.cuh: API declaration for utility functions.
 * 
 * Copyright (c) Michele Collevati
 */


#ifndef UTILS_CUH
#define UTILS_CUH


#include <stdbool.h>
#include <time.h>


/**
 * Global variables
 */


static bool init_PRNG_seed = false;     /**
                                         * Flag to check the initialization of
                                         * the PRNG seed.
                                         */


/**
 * API
 */


/**
 * @brief Wrapper macro to check for errors in runtime CUDA API code. You can
 * wrap each CUDA API call with the gpuErrchk macro, which will process the
 * return status of the CUDA API calls it wraps, for example:
 * 
 * gpuErrchk( cudaMalloc(&a_d, size * sizeof(int)) );
 * 
 * If there is an error in a call, a textual message describing the error and
 * the file and line in your code where the error occured will be emitted to
 * stderr and the application will exit. You could conceivably modify gpuAssert
 * to raise an exception rather than call exit() in a more sophisticated
 * application if it were required.
 *
 * Ref.: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 * 
 * @param [in]ans A CUDA API call.
 * @retval None.
 */
// #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define gpuErrchk(ans) { (ans); }


/**
 * @brief Assert style handler function to check for errors in runtime CUDA API
 * code.
 * 
 * @param [in]code A CUDA error code.
 * @param [in]file A file where the CUDA error occurred.
 * @param [in]line A line in code where the CUDA error occurred.
 * @param [in]abort A flag to choose whether to abort the application or not.
 * @retval None.
 */
inline void gpuAssert(cudaError_t code,
                      const char *file,
                      int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        
        if (abort) {
            exit(code);
        }
    }
}


/**
 * @brief Returns the greater of the given values.
 * 
 * @param [in]a First value.
 * @param [in]b Second value.
 * @retval The greater of a and b.
 */
#undef max
#define max(a,b) (((a) > (b)) ? (a) : (b))


/**
 *  @brief Returns the lesser of the given values.
 * 
 * @param [in]a First value.
 * @param [in]b Second value.
 * @retval The lesser of a and b.
 */
#undef min
#define min(a,b) (((a) > (b)) ? (b) : (a))


/**
 * @brief Initializes the PRNG with a seed.
 * 
 * @retval None.
 */
inline void init_PRNG() {
    if (!init_PRNG_seed) {
        srand(time(NULL));
        init_PRNG_seed = true;
    }
}


/**
 * @brief Finds the index of the maximum int in a device array of positive
 * ints.
 * 
 * @param [in]d_data A device array of positive ints.
 * @param [in]data_len Length of d_data.
 * @retval The index of the maximum int in d_data. -1 if all ints in d_data are
 * negative.
 */
int find_idx_max_int(int *d_data, int data_len);


/**
 * @brief Finds the index of the maximum float in a device array of positive
 * floats.
 * 
 * @param [in]d_data A device array of positive floats.
 * @param [in]data_len Length of d_data.
 * @retval The index of the maximum float in d_data. -1 if all floats in d_data
 * are negative.
 */
int find_idx_max_float(float *d_data, int data_len);


/**
 * @brief Finds the minimum int in a device array of ints.
 * 
 * @param [in]d_data A device array of ints.
 * @param [in]data_len Length of d_data.
 * @retval The minimum int in d_data.
 */
int find_min_int(int *d_data, int data_len);


/**
 * @brief Finds the maximum float in a device array of floats.
 * 
 * @param [in]d_data A device array of floats.
 * @param [in]data_len Length of d_data.
 * @retval The maximum float in d_data.
 */
float find_max_float(float *d_data, int data_len);


/**
 * @brief Initializes or sets int device memory to a int value.
 * Fills the first count ints of the memory area pointed to by devPtr with the
 * constant int value value.
 * Note that this function is asynchronous with respect to the host.
 *
 * @param [out]devPtr A pointer to int device memory.
 * @param [in]value A value to set for each int of specified memory.
 * @param [in]count A number of ints to set.
 * @retval CUDA error types.
 */
__host__ cudaError_t cuda_memset_int(int *devPtr,
                                     int value,
                                     unsigned int count);


/**
 * @brief Initializes or sets float device memory to a float value.
 * Fills the first count floats of the memory area pointed to by devPtr with
 * the constant float value value.
 * Note that this function is asynchronous with respect to the host.
 *
 * @param [out]devPtr A pointer to float device memory.
 * @param [in]value A value to set for each float of specified memory.
 * @param [in]count A number of floats to set.
 * @retval CUDA error types.
 */
__host__ cudaError_t cuda_memset_float(float *devPtr,
                                       float value,
                                       unsigned int count);


#endif
