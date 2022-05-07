/**
 * sat_miracle.cu: definition of the SAT_Miracle API.
 * 
 * Copyright (c) Michele Collevati
 */


#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>


#include "sat_miracle.cuh"


/**
 * API definition
 */


SAT_Miracle *mrc_create_sat_miracle(char *filename, bool gpu) {
    Miracle *mrc = mrc_create_miracle(filename);

    Miracle *d_mrc = NULL;
    if (gpu) {
        d_mrc = mrc_gpu_transfer_miracle_host_to_dev(mrc);
    }

    SAT_Miracle *sat_mrc = (SAT_Miracle *)malloc(sizeof *sat_mrc);

    sat_mrc->mrc = mrc;
    sat_mrc->d_mrc = d_mrc;

    return sat_mrc;
}


void mrc_destroy_sat_miracle(SAT_Miracle *sat_mrc) {
    mrc_destroy_miracle(sat_mrc->mrc);
    mrc_gpu_destroy_miracle(sat_mrc->d_mrc);
    free(sat_mrc);
}


void mrc_print_sat_miracle(SAT_Miracle *sat_mrc) {
    mrc_print_miracle(sat_mrc->mrc);
}


void mrc_sync_sat_miracle(SAT_Miracle *sat_mrc, bool host_to_dev) {
    Miracle *mrc = sat_mrc->mrc;
    Miracle *d_mrc = sat_mrc->d_mrc;

    if (host_to_dev) {
        if (d_mrc != NULL) {
            mrc_gpu_destroy_miracle(d_mrc);
        }

        sat_mrc->d_mrc = mrc_gpu_transfer_miracle_host_to_dev(mrc);
    } else {
        if (mrc != NULL) {
            mrc_destroy_miracle(mrc);
        }

        sat_mrc->mrc = mrc_gpu_transfer_miracle_dev_to_host(d_mrc);
    }
}
