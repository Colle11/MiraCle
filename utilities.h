/*
 * utilities.h
 *
 * file containing helper functions
 *
 */


#ifndef UTILITIES
#define UTILITIES


#include <stdio.h>


/*
 * auxiliary macro function definitions
 */


/*
 * MAX: returns the largest of a and b
 */
#define MAX(a,b)                 \
    ({                           \
        __typeof__ (a) _a = (a); \
        __typeof__ (b) _b = (b); \
        _a > _b ? _a : _b;       \
    })


// /*
//  * NELEMS: determines the number of elements in the array
//  */
// #define NELEMS(x)   (sizeof(x) / sizeof((x)[0]))


/*****************************************************************************/


/*
 * error handling function definitions
 */


static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a )                                        \
    {                                                           \
        if (a == NULL) {                                        \
            printf( "Host memory failed in %s at line %d\n",    \
                    __FILE__, __LINE__ );                       \
            exit( EXIT_FAILURE );                               \
        }                                                       \
    }


/*****************************************************************************/


#endif
