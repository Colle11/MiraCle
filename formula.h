/*
 * formula.h
 *
 * file containing the formula library API
 *
 */


#ifndef FORMULA
#define FORMULA


#include <stdlib.h>


/*
 * auxiliary macro function definitions
 */


/*
 * LIT_TO_IDX: converts from literal to index
 */
#define LIT_TO_IDX(lit, num_vars)                   \
    ({                                              \
        (lit) > 0 ? (lit) : -(lit) + (num_vars);    \
    })


/*
 * IDX_TO_LIT: converts from index to literal
 */
#define IDX_TO_LIT(idx, num_vars)                               \
    ({                                                          \
        (idx) <= (num_vars) ? (idx) : -((idx) - (num_vars));    \
    })


/*
 * LIT_TO_VAR: gets variable from literal
 */
#define LIT_TO_VAR(lit)   (abs(lit))


/*
 * IDX_TO_VAR: gets variable from index
 */
#define IDX_TO_VAR(idx, num_vars)                           \
    ({                                                      \
        (idx) <= (num_vars) ? (idx) : (idx) - (num_vars);   \
    })


/*****************************************************************************/


// static limits to formula size
#define MAX_NUM_CLAUSES 550000      // maximum number of clauses
#define MAX_NUM_LITS    8000000     // maximum number of literals
#define MAX_NUM_VARS    3500        // maximum number of variables


/*****************************************************************************/


/*
 * definition of the formula data type
 */
typedef struct formula {

    /*
     * For convenience, one-based indexing is used. It follows that the length
     * of the arrays is incremented by one since the zero index is not used.
     */

    unsigned int dec_lvl;   // decision level (dynamic)

    /*
     * variables
     */
    unsigned int s_num_vars;        // number of variables (static)
    unsigned int d_num_vars;        // number of variables NOT yet assigned (dynamic)
    int var_ass[MAX_NUM_VARS + 1];  /* variable assignments (dynamic)
                                     * var_ass[i] = ... :
                                     * 1. 0         if variable i has NOT yet been
                                     *              assigned
                                     * 2. l > 0     if variable i has been POSITIVELY
                                     *              assigned at level l
                                     * 3. -l < 0    if variable i has been NEGATIVELY
                                     *              assigned at level l
                                     */
    
    /*
     * clauses
     */
    unsigned int s_num_clauses;     // number of clauses (static)
    unsigned int d_num_clauses;     // number of clauses NOT yet satisfied (dynamic)
    unsigned int clause_sat[MAX_NUM_CLAUSES + 1];   /* satisfiability of clauses
                                                     * (dynamic)
                                                     * clause_sat[i] = ... :
                                                     * 1. 0         if clause i
                                                     *              has NOT yet
                                                     *              been satisfied
                                                     * 2. l > 0     if clause i
                                                     *              has been satisfied
                                                     *              at level l
                                                     */
    unsigned int s_clause_len[MAX_NUM_CLAUSES + 1]; // clause lengths (static)
    unsigned int d_clause_len[MAX_NUM_CLAUSES + 1]; // clause lengths (dynamic)

    /*
     * literals
     */
    unsigned int s_num_lits;    // number of literals (static)
    unsigned int d_num_lits;    // number of literals still active (dynamic)
    unsigned int s_occ_lits[MAX_NUM_VARS * 2 + 1];  /* number of occurrences of
                                                     * literals (static)
                                                     */
    unsigned int d_occ_lits[MAX_NUM_VARS * 2 + 1];  /* number of occurrences of
                                                     * literals (dynamic)
                                                     */
    
    /*
     * Compressed Sparse Column Format (CSC)
     */
    unsigned int csc_row_ind[MAX_NUM_LITS + 1];     /* occurrences of clauses ordered
                                                     * by literal (static)
                                                     */
    unsigned int csc_col_ptr[MAX_NUM_VARS * 2 + 2];     /* indices into the array
                                                         * csc_row_ind.
                                                         * Indices of the first clause
                                                         * in the i-th literal for
                                                         * i=1,...,s_num_vars*2 (static)
                                                         */
    
    /*
     * Compressed Sparse Row Format (CSR)
     */
    unsigned int csr_col_ind[MAX_NUM_LITS + 1];     /* occurrences of literals ordered
                                                     * by clause (static)
                                                     */
    unsigned int csr_row_ptr[MAX_NUM_CLAUSES + 2];      /* indices into the array
                                                         * csr_col_ind.
                                                         * Indices of the first literal
                                                         * in the i-th clause for
                                                         * i=1,...,s_num_clauses (static)
                                                         */

} formula;


/*****************************************************************************/


/*
 * global variables
 */

extern formula *phi;


/*****************************************************************************/


/*
 * API function prototypes
 */

int new_formula(char *filename);
void delete_formula();
void print_formula();
void new_lit_assignments(int *new_lit_ass, unsigned int new_lit_ass_len);
void backjumping(unsigned int bj_dec_lvl);


/*****************************************************************************/


#endif
