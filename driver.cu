/*
 * driver.cu
 *
 * file containing the main of the program
 *
 */


#include <stdio.h>


#include "miracle.h"


// number of program parameters
#define NUM_PARAMS 2


int main(int argc, char *argv[]) {
    if(argc != NUM_PARAMS) {
        fprintf(stderr, "usage: %s filename\n", argv[0]);
        return 1;
    }

    char *filename = argv[1];
    int err = new_miracle(filename);
    
    if(err) {
        return err;
    }

    // DEBUG
    printf("DLIS = %i\n", DLIS_heuristic());
    printf("RDLIS = %i\n", RDLIS_heuristic());
    printf("DLCS = %i\n", DLCS_heuristic());
    printf("RDLCS = %i\n", RDLCS_heuristic());
    printf("JW_OS = %i\n", JW_OS_heuristic());
    printf("JW_TS = %i\n", JW_TS_heuristic());
    printf("MOM = %i\n", MOM_heuristic(3));
    printf("BOHM = %i\n", BOHM_heuristic(1, 2));
    printf("RAND = %i\n", RAND_heuristic());
    // END DEBUG

    delete_miracle();

    return 0;
}
