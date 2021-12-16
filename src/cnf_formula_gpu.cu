/**
 * cnf_formula_gpu.cu: definition of the CNF_Formula device API.
 * 
 * Copyright (c) Michele Collevati
 */


#include <stdlib.h>
#include <stdio.h>


#include "cnf_formula_gpu.cuh"
#include "utils.cuh"


/**
 * API definition
 */


CNF_Formula *cnf_gpu_transfer_formula_host_to_dev(CNF_Formula *phi) {
    CNF_Formula *d_phi;
    gpuErrchk( cudaMalloc((void**)&d_phi, sizeof *d_phi) );
    gpuErrchk( cudaMemcpy(d_phi, phi, sizeof *d_phi, cudaMemcpyHostToDevice) );
    
    Lidx *d_clauses;
    gpuErrchk( cudaMalloc((void**)&d_clauses,
                          sizeof *d_clauses * phi->clauses_len) );
    gpuErrchk( cudaMemcpy(&(d_phi->clauses), &d_clauses,
                          sizeof d_clauses,
                          cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_clauses, phi->clauses,
                          sizeof *d_clauses * phi->clauses_len,
                          cudaMemcpyHostToDevice) );
    
    int *d_clause_indices;
    gpuErrchk( cudaMalloc((void**)&d_clause_indices,
                          sizeof *d_clause_indices *
                          phi->clause_indices_len) );
    gpuErrchk( cudaMemcpy(&(d_phi->clause_indices), &d_clause_indices,
                          sizeof d_clause_indices,
                          cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_clause_indices, phi->clause_indices,
                          sizeof *d_clause_indices * phi->clause_indices_len,
                          cudaMemcpyHostToDevice) );

    return d_phi;
}


CNF_Formula *cnf_gpu_transfer_formula_dev_to_host(CNF_Formula *d_phi) {
    CNF_Formula *phi = (CNF_Formula *)malloc(sizeof *phi);
    gpuErrchk( cudaMemcpy(phi, d_phi, sizeof *phi, cudaMemcpyDeviceToHost) );

    phi->clauses = (Lidx *)malloc(sizeof *(phi->clauses) * phi->clauses_len);
    Lidx *d_clauses;
    gpuErrchk( cudaMemcpy(&d_clauses, &(d_phi->clauses),
                          sizeof d_clauses,
                          cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(phi->clauses, d_clauses,
                          sizeof *d_clauses * phi->clauses_len,
                          cudaMemcpyDeviceToHost) );

    phi->clause_indices = (int *)malloc(sizeof *(phi->clause_indices) *
                                        phi->clause_indices_len);
    int *d_clause_indices;
    gpuErrchk( cudaMemcpy(&d_clause_indices, &(d_phi->clause_indices),
                          sizeof d_clause_indices,
                          cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(phi->clause_indices, d_clause_indices,
                          sizeof *d_clause_indices * phi->clause_indices_len,
                          cudaMemcpyDeviceToHost) );

    return phi;
}


void cnf_gpu_destroy_formula(CNF_Formula *d_phi) {
    Lidx *d_clauses;
    gpuErrchk( cudaMemcpy(&d_clauses, &(d_phi->clauses),
                          sizeof d_clauses,
                          cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaFree(d_clauses) );

    int *d_clause_indices;
    gpuErrchk( cudaMemcpy(&d_clause_indices, &(d_phi->clause_indices),
                          sizeof d_clause_indices,
                          cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaFree(d_clause_indices) );

    gpuErrchk( cudaFree(d_phi) );
}
