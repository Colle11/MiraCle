/**
 * cnf_formula_gpu.cuh: declaration of the CNF_Formula device API.
 * 
 * Copyright (c) Michele Collevati
 */


#ifndef CNF_FORMULA_GPU_CUH
#define CNF_FORMULA_GPU_CUH


#include "cnf_formula.cuh"


/**
 * API
 */


/**
 * @brief Transfers a formula from the host to the device.
 * 
 * @param [in]phi A host formula.
 * @retval A device formula.
 */
CNF_Formula *cnf_gpu_transfer_formula_host_to_dev(CNF_Formula *phi);


/**
 * @brief Transfers a formula from the device to the host.
 * 
 * @param [in]d_phi A device formula.
 * @retval A host formula.
 */
CNF_Formula *cnf_gpu_transfer_formula_dev_to_host(CNF_Formula *d_phi);


/**
 * @brief Destroys a device formula.
 * 
 * @param [in]d_phi A device formula.
 * @retval None.
 */
void cnf_gpu_destroy_formula(CNF_Formula *d_phi);


#endif
