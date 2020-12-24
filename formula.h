/*
 * formula.h
 *
 * file containing the formula data type and its API
 */

#ifndef FORMULA
#define FORMULA

// static limits to formula size
#define MAX_NUM_CLAUSES 550000  // maximum number of clauses
#define MAX_NUM_LITS 8000000    // maximum number of literals
#define MAX_NUM_VARS 1000       // maximum number of variables

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
    unsigned int d_num_lits;    // number of literals NOT yet assigned (dynamic)
    unsigned int s_occ_lits[MAX_NUM_VARS * 2 + 1];  /* number of occurrences of
                                                     * literals (static)
                                                     */
    unsigned int d_occ_lits[MAX_NUM_VARS * 2 + 1];  /* number of occurrences of
                                                     * literals (dynamic)
                                                     */
    
    /*
     * Compressed Sparse Column Format (CSC)
     */
    unsigned int csc_row_ind_f[MAX_NUM_LITS + 1];   /* occurrences of clauses ordered
                                                     * by literal (static)
                                                     */
    unsigned int csc_col_ptr_f[MAX_NUM_VARS * 2 + 2];   /* indices into the array
                                                         * csc_row_ind_f.
                                                         * Indices of the first clause
                                                         * in the i-th literal for
                                                         * i=1,...,s_num_vars*2 (static)
                                                         */
    
    /*
     * Compressed Sparse Row Format (CSR)
     */
    unsigned int csr_col_ind_f[MAX_NUM_LITS + 1];   /* occurrences of literals ordered
                                                     * by clause (static)
                                                     */
    unsigned int csr_row_ptr_f[MAX_NUM_CLAUSES + 2];    /* indices into the array
                                                         * csr_col_ind_f.
                                                         * Indices of the first literal
                                                         * in the i-th clause for
                                                         * i=1,...,s_num_clauses (static)
                                                         */

} formula;

#endif
