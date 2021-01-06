/*
 * miracle.cu
 *
 * file containing the MiraCle library
 *
 */


#include "miracle.h"
#include "formula.h"


/*
 * API function prototypes
 */

int new_miracle(char *filename);
void delete_miracle();


/*****************************************************************************/


/*
 * auxiliary function prototypes
 */


/*****************************************************************************/


/*
 * API function definitions
 */

/*
 * new_miracle: creates and initializes the oracle MiraCle
 */
int new_miracle(char *filename) {
    int err = new_formula(filename);
    
    if(err) {
        return err;
    }

    print_formula();

    return 0;
}

/*
 * delete_miracle: destroys the oracle MiraCle previously allocated by new_miracle
 */
void delete_miracle() {
    delete_formula();
}


/*****************************************************************************/


/*
 * auxiliary function definitions
 */


/*****************************************************************************/
