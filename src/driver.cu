/**
 * driver.cu: program to test MiraCle.
 * 
 * Copyright (c) Michele Collevati
 */


#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>


#include "sat_miracle.cuh"
#include "miracle_dynamic.cuh"
#include "launch_parameters_gpu.cuh"


#define NUM_ARGS (1)    // Number of program arguments.


int main(int argc, char *argv[]) {
    char *prog_name = argv[0];      // Program name.

    if ((argc - 1) != NUM_ARGS) {
        fprintf(stderr, "usage: %s filename\n", prog_name);
        exit(EXIT_FAILURE);
    }

    char *filename = argv[1];

    // Constants of weight functions.
    const int POSIT_n = 8;
    const int BOHM_alpha = 1;
    const int BOHM_beta = 2;

    // Assigned literals.
    int lits_len = 1;
    Lit lits[lits_len];
    for (int l = 0; l < lits_len; l++) {
        lits[l] = l + 1;
    }

    // Testing MiraCle (serial version).
    printf("**************************************************************\n");
    printf("***********    Testing MiraCle (serial version)    ***********\n");
    printf("**************************************************************\n");
    printf("\n");

    SAT_Miracle *sat_mrc = mrc_create_sat_miracle(filename, true);

    mrc_print_sat_miracle(sat_mrc);

    mrc_assign_lits(lits, lits_len, sat_mrc);

    Lit RAND_blit = mrc_RAND_heuristic(sat_mrc);
    Lit JW_OS_blit = mrc_JW_OS_heuristic(sat_mrc);
    Lit JW_TS_blit = mrc_JW_TS_heuristic(sat_mrc);
    Lit BOHM_blit = mrc_BOHM_heuristic(sat_mrc, BOHM_alpha, BOHM_beta);
    Lit POSIT_blit = mrc_POSIT_heuristic(sat_mrc, POSIT_n);
    Lit DLIS_blit = mrc_DLIS_heuristic(sat_mrc);
    Lit DLCS_blit = mrc_DLCS_heuristic(sat_mrc);
    Lit RDLIS_blit = mrc_RDLIS_heuristic(sat_mrc);
    Lit RDLCS_blit = mrc_RDLCS_heuristic(sat_mrc);
    
    printf("RAND branching literal = %d\n", RAND_blit);
    printf("JW-OS branching literal = %d\n", JW_OS_blit);
    printf("JW-TS branching literal = %d\n", JW_TS_blit);
    printf("BOHM branching literal = %d\n", BOHM_blit);
    printf("POSIT branching literal = %d\n", POSIT_blit);
    printf("DLIS branching literal = %d\n", DLIS_blit);
    printf("DLCS branching literal = %d\n", DLCS_blit);
    printf("RDLIS branching literal = %d\n", RDLIS_blit);
    printf("RDLCS branching literal = %d\n", RDLCS_blit);
    
    printf("\n");
    printf("**************************************************************\n");
    printf("*********    End testing MiraCle (serial version)    *********\n");
    printf("**************************************************************\n");
    // End testing MiraCle (serial version).

    printf("\n");

    // Testing Dynamic MiraCle (serial version).
    printf("**************************************************************\n");
    printf("*******    Testing Dynamic MiraCle (serial version)    *******\n");
    printf("**************************************************************\n");
    printf("\n");

    Miracle_Dyn *mrc_dyn = mrc_dyn_create_miracle(filename);
    
    mrc_dyn_assign_lits(lits, lits_len, mrc_dyn);
    
    Lit RAND_blit_dyn = mrc_dyn_RAND_heuristic(mrc_dyn);
    Lit JW_OS_blit_dyn = mrc_dyn_JW_OS_heuristic(mrc_dyn);
    Lit JW_TS_blit_dyn = mrc_dyn_JW_TS_heuristic(mrc_dyn);
    Lit BOHM_blit_dyn = mrc_dyn_BOHM_heuristic(mrc_dyn, BOHM_alpha, BOHM_beta);
    Lit POSIT_blit_dyn = mrc_dyn_POSIT_heuristic(mrc_dyn, POSIT_n);
    Lit DLIS_blit_dyn = mrc_dyn_DLIS_heuristic(mrc_dyn);
    Lit DLCS_blit_dyn = mrc_dyn_DLCS_heuristic(mrc_dyn);
    Lit RDLIS_blit_dyn = mrc_dyn_RDLIS_heuristic(mrc_dyn);
    Lit RDLCS_blit_dyn = mrc_dyn_RDLCS_heuristic(mrc_dyn);
    
    printf("RAND branching literal dynamic = %d\n", RAND_blit_dyn);
    printf("JW-OS branching literal dynamic = %d\n", JW_OS_blit_dyn);
    printf("JW-TS branching literal dynamic = %d\n", JW_TS_blit_dyn);
    printf("BOHM branching literal dynamic = %d\n", BOHM_blit_dyn);
    printf("POSIT branching literal dynamic = %d\n", POSIT_blit_dyn);
    printf("DLIS branching literal dynamic = %d\n", DLIS_blit_dyn);
    printf("DLCS branching literal dynamic = %d\n", DLCS_blit_dyn);
    printf("RDLIS branching literal dynamic = %d\n", RDLIS_blit_dyn);
    printf("RDLCS branching literal dynamic = %d\n", RDLCS_blit_dyn);
    
    mrc_dyn_destroy_miracle(mrc_dyn);

    printf("\n");
    printf("**************************************************************\n");
    printf("*****    End testing Dynamic MiraCle (serial version)    *****\n");
    printf("**************************************************************\n");
    // End testing Dynamic MiraCle (serial version).

    printf("\n");

    // Testing MiraCle (parallel version).
    printf("**************************************************************\n");
    printf("**********    Testing MiraCle (parallel version)    **********\n");
    printf("**************************************************************\n");
    printf("\n");

    mrc_gpu_assign_lits(lits, lits_len, sat_mrc);

    // mrc_sync_sat_miracle(sat_mrc, true);

    Lit JW_OS_blit_gpu = mrc_gpu_JW_OS_heuristic(sat_mrc);
    Lit JW_TS_blit_gpu = mrc_gpu_JW_TS_heuristic(sat_mrc);
    Lit BOHM_blit_gpu = mrc_gpu_BOHM_heuristic(sat_mrc, BOHM_alpha, BOHM_beta);
    Lit POSIT_blit_gpu = mrc_gpu_POSIT_heuristic(sat_mrc, POSIT_n);
    Lit DLIS_blit_gpu = mrc_gpu_DLIS_heuristic(sat_mrc);
    Lit DLCS_blit_gpu = mrc_gpu_DLCS_heuristic(sat_mrc);
    Lit RDLIS_blit_gpu = mrc_gpu_RDLIS_heuristic(sat_mrc);
    Lit RDLCS_blit_gpu = mrc_gpu_RDLCS_heuristic(sat_mrc);
    
    printf("JW-OS branching literal GPU = %d\n", JW_OS_blit_gpu);
    printf("JW-TS branching literal GPU = %d\n", JW_TS_blit_gpu);
    printf("BOHM branching literal GPU = %d\n", BOHM_blit_gpu);
    printf("POSIT branching literal GPU = %d\n", POSIT_blit_gpu);
    printf("DLIS branching literal GPU = %d\n", DLIS_blit_gpu);
    printf("DLCS branching literal GPU = %d\n", DLCS_blit_gpu);
    printf("RDLIS branching literal GPU = %d\n", RDLIS_blit_gpu);
    printf("RDLCS branching literal GPU = %d\n", RDLCS_blit_gpu);
    
    mrc_destroy_sat_miracle(sat_mrc);
    
    printf("\n");
    printf("**************************************************************\n");
    printf("********    End testing MiraCle (parallel version)    ********\n");
    printf("**************************************************************\n");
    // End testing MiraCle (parallel version).

    exit(EXIT_SUCCESS);
}
