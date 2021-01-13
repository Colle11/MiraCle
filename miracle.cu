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
    
    // DEBUG
    int new_lit_ass_fst[] = {1};
    new_lit_assignments(new_lit_ass_fst, 1);
    int new_lit_ass_snd[] = {-3};
    new_lit_assignments(new_lit_ass_snd, 1);
    int new_lit_ass_trd[] = {33, -18, -45, 94, 58, 56, -65, 30, -48, -77};
    new_lit_assignments(new_lit_ass_trd, 10);
    backjumping(1);
    int new_lit_ass_frt[] = {-72, 20, 68, -54, 59, -57, 89, 98, -96, 83};
    new_lit_assignments(new_lit_ass_frt, 10);
    backjumping(0);
    // END DEBUG

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
