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

    delete_miracle();

    return 0;
}
