/**
 * miracle_gpu.cuh: declaration of the Miracle device API.
 * 
 * Copyright (c) Michele Collevati
 */


#ifndef MIRACLE_GPU_CUH
#define MIRACLE_GPU_CUH


#include "miracle.cuh"


/**
 * API
 */


/**
 * @brief Transfers a miracle from the host to the device.
 * 
 * @param [in]mrc A host miracle.
 * @retval A device miracle.
 */
Miracle *mrc_gpu_transfer_miracle_host_to_dev(Miracle *mrc);


/**
 * @brief Transfers a miracle from the device to the host.
 * 
 * @param [in]d_mrc A device miracle.
 * @retval A host miracle.
 */
Miracle *mrc_gpu_transfer_miracle_dev_to_host(Miracle *d_mrc);


/**
 * @brief Destroys a device miracle.
 * 
 * @param [in]d_mrc A device miracle.
 * @retval None.
 */
void mrc_gpu_destroy_miracle(Miracle *d_mrc);


/**
 * @brief Assigns literals in a device miracle.
 * 
 * @param [in]lits An array of assigned literals.
 * @param [in]lits_len Length of lits, which is the number of assigned
 * literals.
 * @param [in/out]d_mrc A device miracle.
 * @retval None.
 */
void mrc_gpu_assign_lits(Lit *lits, int lits_len, Miracle *d_mrc);


/**
 * @brief Increases the decision level in a device miracle.
 * 
 * @param [in/out]d_mrc A device miracle.
 * @retval None.
 */
void mrc_gpu_increase_decision_level(Miracle *d_mrc);


/**
 * @brief Backjumps to a decision level in a device miracle.
 * 
 * @param [in]bj_dec_lvl A backjump decision level. A bj_dec_lvl < 1 resets the
 * device miracle.
 * @param [in/out]d_mrc A device miracle.
 * @retval None.
 */
void mrc_gpu_backjump(int bj_dec_lvl, Miracle *d_mrc);


/**
 * @brief Computes the JW-OS heuristic on the device.
 * 
 * @param [in]d_mrc A device miracle.
 * @retval The branching literal.
 */
Lit mrc_gpu_JW_OS_heuristic(Miracle *d_mrc);


/**
 * @brief Computes the JW-TS heuristic on the device.
 * 
 * @param [in]d_mrc A device miracle.
 * @retval The branching literal.
 */
Lit mrc_gpu_JW_TS_heuristic(Miracle *d_mrc);


/**
 * @brief Computes the BOHM heuristic on the device.
 *
 * @param [in]d_mrc A device miracle.
 * @param [in]alpha A constant of the BOHM weight function.
 * @param [in]beta A constant of the BOHM weight function.
 * @retval The branching literal.
 */
Lit mrc_gpu_BOHM_heuristic(Miracle *d_mrc, const int alpha, const int beta);


/**
 * @brief Computes the POSIT heuristic on the device.
 * 
 * @param [in]d_mrc A device miracle.
 * @param [in]n A constant of the POSIT weight function.
 * @retval The branching literal.
 */
Lit mrc_gpu_POSIT_heuristic(Miracle *d_mrc, const int n);


/**
 * @brief Computes the DLIS heuristic on the device.
 * 
 * @param [in]d_mrc A device miracle.
 * @retval The branching literal.
 */
Lit mrc_gpu_DLIS_heuristic(Miracle *d_mrc);


/**
 * @brief Computes the DLCS heuristic on the device.
 * 
 * @param [in]d_mrc A device miracle.
 * @retval The branching literal.
 */
Lit mrc_gpu_DLCS_heuristic(Miracle *d_mrc);


/**
 * @brief Computes the RDLIS heuristic on the device.
 * 
 * @param [in]d_mrc A device miracle.
 * @retval The branching literal.
 */
Lit mrc_gpu_RDLIS_heuristic(Miracle *d_mrc);


/**
 * @brief Computes the RDLCS heuristic on the device.
 * 
 * @param [in]d_mrc A device miracle.
 * @retval The branching literal.
 */
Lit mrc_gpu_RDLCS_heuristic(Miracle *d_mrc);


#endif
