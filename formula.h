/*
 * formula.h
 *
 * file containing the formula data type and its API
 */

#ifndef FORMULA
#define FORMULA

// static limits to formula size
#define MAX_NUM_CLAUSES 550000
#define MAX_NUM_LITS 8000000
#define MAX_NUM_VARS 1000

/*
 * definition of the formula data type
 */
typedef struct formula {

    /*
     * variables
     */
    unsigned int s_num_vars;    // number of variables (static)
    unsigned int d_num_vars;    // number of variables NOT yet assigned (dynamic)
    int var_ass[MAX_NUM_VARS];  /* variable assignments (dynamic)
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
    unsigned int clause_sat[MAX_NUM_CLAUSES];   /* satisfiability of clauses (dynamic)
                                                 * clause_sat[i] = ... :
                                                 * 1. 0         if clause i has NOT
                                                 *              yet been satisfied
                                                 * 2. l > 0     if clause i has been
                                                 *              satisfied at level l
                                                 */
    unsigned int s_clause_len[MAX_NUM_CLAUSES]; // clause lengths (static)
    unsigned int d_clause_len[MAX_NUM_CLAUSES]; // clause lengths (dynamic)

    /*
     * literals
     */
    unsigned int s_num_lits;    // number of literals (static)
    unsigned int d_num_lits;    // number of literals NOT yet assigned (dynamic)
    unsigned int s_occ_lits[MAX_NUM_VARS * 2];  /* number of occurrences of literals
                                                 * (static)
                                                 */
    unsigned int d_occ_lits[MAX_NUM_VARS * 2];  /* number of occurrences of literals
                                                 * (dynamic)
                                                 */
} formula;

#endif
