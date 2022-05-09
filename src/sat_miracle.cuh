/**
 * sat_miracle.cuh: definition of the SAT_Miracle data type.
 * 
 * Copyright (c) Michele Collevati
 */


#ifndef SAT_MIRACLE_CUH
#define SAT_MIRACLE_CUH


#include <stdbool.h>


#include "miracle.cuh"
#include "miracle_gpu.cuh"


/**
 * @brief SAT_Miracle data type.
 */
typedef struct sat_miracle {
    Miracle *mrc;               // Host miracle.
    Miracle *d_mrc;             // Device miracle.

    int num_blks_num_vars;      /**
                                 * Number of blocks per kernel launch based on
                                 * the number of variables.
                                 */
    int num_blks_num_clauses;   /**
                                 * Number of blocks per kernel launch based on
                                 * the number of clauses.
                                 */
    int num_thds_per_blk;       // Number of threads per block.
} SAT_Miracle;


/**
 * API
 */


/**
 * @brief Creates a sat_miracle.
 * 
 * @param [in]filename Filename of a formula in DIMACS CNF format.
 * @param [in]gpu A flag to initialize the device miracle or not.
 * @retval An initialized sat_miracle.
 */
SAT_Miracle *mrc_create_sat_miracle(char *filename, bool gpu);


/**
 * @brief Destroys a sat_miracle.
 * 
 * @param [in]sat_mrc A sat_miracle.
 * @retval None.
 */
void mrc_destroy_sat_miracle(SAT_Miracle *sat_mrc);


/**
 * @brief Prints a sat_miracle.
 * 
 * @param [in]sat_mrc A sat_miracle.
 * @retval None.
 */
void mrc_print_sat_miracle(SAT_Miracle *sat_mrc);


/**
 * @brief Synchronizes the host and device miracle.
 * 
 * @param [in/out]sat_mrc A sat_miracle.
 * @param [in]host_to_dev A flag to synchronize the miracle from host to device
 * or vice versa.
 * @retval None.
 */
void mrc_sync_sat_miracle(SAT_Miracle *sat_mrc, bool host_to_dev);


#endif
