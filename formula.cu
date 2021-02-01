/*
 * formula.cu
 *
 * file containing the formula library
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>


#include "formula.h"
#include "utilities.h"


/*
 * global variables
 */

formula *phi;


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


/*
 * auxiliary function prototypes
 */

static void alloc_formula();
static void free_formula();
static int parse_formula(char *filename);
static void skip_line(FILE *fp);
static void parse_info(
                       FILE *fp,
                       unsigned int *p_num_vars,
                       unsigned int *p_num_clauses
                      );
static void parse_clause(
                         FILE *fp,
                         char c,
                         bool (*f_mat)[MAX_NUM_CLAUSES + 1][MAX_NUM_VARS * 2 + 1],
                         unsigned int *num_vars,
                         unsigned int *num_lits,
                         unsigned int *num_clauses,
                         unsigned int *p_num_vars
                        );
static void parse_lit(
                      FILE *fp,
                      char c,
                      bool (*f_mat)[MAX_NUM_CLAUSES + 1][MAX_NUM_VARS * 2 + 1],
                      unsigned int *num_vars,
                      unsigned int *num_lits,
                      unsigned int *num_clauses,
                      unsigned int *p_num_vars
                     );
static void init_formula(
                         bool (*f_mat)[MAX_NUM_CLAUSES + 1][MAX_NUM_VARS * 2 + 1],
                         unsigned int num_vars,
                         unsigned int num_lits,
                         unsigned int num_clauses
                        );
static void init_csc(
                     bool (*f_mat)[MAX_NUM_CLAUSES + 1][MAX_NUM_VARS * 2 + 1],
                     unsigned int num_vars,
                     unsigned int num_clauses
                    );
static void init_csr(
                     bool (*f_mat)[MAX_NUM_CLAUSES + 1][MAX_NUM_VARS * 2 + 1],
                     unsigned int num_vars,
                     unsigned int num_clauses
                    );
static void init_vars(unsigned int num_vars);
static void init_clauses(unsigned int num_clauses);
static void init_lits(unsigned int num_vars, unsigned int num_lits);
static void ass_lit_updates(int *new_lit_ass, unsigned int new_lit_ass_len);
static void neg_lit_updates(int *new_lit_ass, unsigned int new_lit_ass_len);
static void ass_lit_restores(int *lits_restore, unsigned int lits_restore_len, unsigned int bj_dec_lvl);
static void neg_lit_restores(int *lits_restore, unsigned int lits_restore_len);


/*****************************************************************************/


/*
 * API function definitions
 */

/*
 * new_formula: creates and initializes the formula from a file in DIMACS .cnf format
 */
int new_formula(char *filename) {
    alloc_formula();

    return parse_formula(filename);
}


/*
 * delete_formula: destroys the formula previously allocated by new_formula
 */
void delete_formula() {
    free_formula();
}


/*
 * print_formula: prints the formula data structure
 */
void print_formula() {
    /*
     * decision level
     */
    printf("******************\n");
    printf("* decision level *\n");
    printf("******************\n");
    printf("decision level (dynamic): %u\n\n\n", phi->dec_lvl);


    /*
     * variables
     */
    printf("*************\n");
    printf("* variables *\n");
    printf("*************\n");
    printf("number of variables (static): %u\n\n", phi->s_num_vars);
    printf("number of variables NOT yet assigned (dynamic): %u\n\n", phi->d_num_vars);
    printf("variable assignments (dynamic): ");
    for(unsigned int v = 1; v <= phi->s_num_vars; v++) {
        printf("%i ", phi->var_ass[v]);
    }
    printf("\n\n\n");


    /*
     * clauses
     */
    printf("***********\n");
    printf("* clauses *\n");
    printf("***********\n");
    printf("number of clauses (static): %u\n\n", phi->s_num_clauses);
    printf("number of clauses NOT yet satisfied (dynamic): %u\n\n", phi->d_num_clauses);
    printf("satisfiability of clauses (dynamic): ");
    for(unsigned int c = 1; c <= phi->s_num_clauses; c++) {
        printf("%u ", phi->clause_sat[c]);
    }
    printf("\n\n");
    printf("clause lengths (static): ");
    for(unsigned int c = 1; c <= phi->s_num_clauses; c++) {
        printf("%u ", phi->s_clause_len[c]);
    }
    printf("\n\n");
    printf("clause lengths (dynamic): ");
    for(unsigned int c = 1; c <= phi->s_num_clauses; c++) {
        printf("%u ", phi->d_clause_len[c]);
    }
    printf("\n\n\n");


    /*
     * literals
     */
    printf("************\n");
    printf("* literals *\n");
    printf("************\n");
    printf("number of literals (static): %u\n\n", phi->s_num_lits);
    printf("number of literals still active (dynamic): %u\n\n", phi->d_num_lits);
    printf("number of occurrences of literals (static): ");
    for(unsigned int l = 1; l <= phi->s_num_vars * 2; l++) {
        printf("%u ", phi->s_occ_lits[l]);
    }
    printf("\n\n");
    printf("number of occurrences of literals (dynamic): ");
    for(unsigned int l = 1; l <= phi->s_num_vars * 2; l++) {
        printf("%u ", phi->d_occ_lits[l]);
    }
    printf("\n\n\n");


    /*
     * Compressed Sparse Column Format (CSC)
     */
    printf("*****************************************\n");
    printf("* Compressed Sparse Column Format (CSC) *\n");
    printf("*****************************************\n");
    printf("csc_row_ind: ");
    for(unsigned int l = 1; l <= phi->s_num_lits; l++) {
        printf("%u ", phi->csc_row_ind[l]);
    }
    printf("\n\n");
    printf("csc_col_ptr: ");
    for(unsigned int l = 1; l <= phi->s_num_vars * 2 + 1; l++) {
        printf("%u ", phi->csc_col_ptr[l]);
    }
    printf("\n\n\n");

    
    /*
     * Compressed Sparse Row Format (CSR)
     */
    printf("**************************************\n");
    printf("* Compressed Sparse Row Format (CSR) *\n");
    printf("**************************************\n");
    printf("csr_col_ind: ");
    for(unsigned int l = 1; l <= phi->s_num_lits; l++) {
        printf("%u ", phi->csr_col_ind[l]);
    }
    printf("\n\n");
    printf("csr_row_ptr: ");
    for(unsigned int c = 1; c <= phi->s_num_clauses + 1; c++) {
        printf("%u ", phi->csr_row_ptr[c]);
    }
    printf("\n");
}


/*
 * new_lit_assignments: updates the formula data structure with the new literal
 *                      assignments
 */
void new_lit_assignments(int *new_lit_ass, unsigned int new_lit_ass_len) {
    // updates the decision level
    phi->dec_lvl++;

    // assigned literal updates
    ass_lit_updates(new_lit_ass, new_lit_ass_len);

    // negated literal updates
    neg_lit_updates(new_lit_ass, new_lit_ass_len);
}


/*
 * backjumping: restores the formula data structure to the backjumping decision
 *              level
 */
void backjumping(unsigned int bj_dec_lvl) {
    // retrieves the assigned literals to be restored
    int *lits_restore;
    HANDLE_ERROR( cudaHostAlloc( (void**) &lits_restore,
                                 sizeof( *lits_restore ) * MAX_NUM_VARS,
                                 cudaHostAllocDefault ) );
    unsigned int lits_restore_len = 0;

    for(unsigned int v = 1; v <= phi->s_num_vars; v++) {
        int v_ass = phi->var_ass[v];
        unsigned int v_dec_lvl = abs(v_ass);

        if(v_dec_lvl > bj_dec_lvl) {
            if(v_ass > 0) {
                lits_restore[lits_restore_len] = v;
            } else {    // v_ass < 0
                lits_restore[lits_restore_len] = -v;
            }

            lits_restore_len++;
        }
    }

    // negated literal restores
    neg_lit_restores(lits_restore, lits_restore_len);

    // assigned literal restores
    ass_lit_restores(lits_restore, lits_restore_len, bj_dec_lvl);

    // restores the decision level
    phi->dec_lvl = bj_dec_lvl;
}


/*****************************************************************************/


/*
 * auxiliary function definitions
 */

/*
 * alloc_formula: allocates the empty formula on the host memory
 */
static void alloc_formula() {
    HANDLE_ERROR( cudaHostAlloc( (void**) &phi,
                                 sizeof( *phi ),
                                 cudaHostAllocDefault ) );
}


/*
 * free_formula: deallocates the formula on the host memory
 */
static void free_formula() {
    HANDLE_ERROR( cudaFreeHost( phi ) );
}


/*
 * parse_formula: parses the formula contained in a file in DIMACS .cnf format
 */
static int parse_formula(char *filename) {
    FILE *fp = fopen(filename, "r");

    if(fp == NULL) {
        fprintf(stderr, "Error opening file %s.\n", filename);
        return 1;
    }

    // clause-literal matrix
    bool (*f_mat)[MAX_NUM_CLAUSES + 1][MAX_NUM_VARS * 2 + 1];
    HANDLE_ERROR( cudaHostAlloc( (void**) &f_mat,
                                 sizeof( *f_mat ),
                                 cudaHostAllocDefault ) );
    char c;
    unsigned int p_num_vars;
    unsigned int p_num_clauses;
    unsigned int num_vars = 0;
    unsigned int num_lits = 0;
    unsigned int num_clauses = 0;

    while(!feof(fp)) {
        c = fgetc(fp);

        if(c == 'c') {
            skip_line(fp);      // skip a comment line
        } else if(c == 'p') {
            parse_info(fp, &p_num_vars, &p_num_clauses);    // parse the info line
        } else if(('1' <= c && c <= '9') || c == '-') {
            // parse a clause line
            parse_clause(fp, c, f_mat, &num_vars, &num_lits, &num_clauses, &p_num_vars);
        }
    }

    fclose(fp);

    // file consistency checking
    if(p_num_vars != num_vars || p_num_clauses != num_clauses) {
        fprintf(stderr, "The %s file in DIMACS .cnf format is inconsistent.\n",
                filename);
        return 1;
    }

    init_formula(f_mat, num_vars, num_lits, num_clauses);

    HANDLE_ERROR( cudaFreeHost( f_mat ) );

    return 0;
}


/*
 * skip_line: skips a line in a file
 */
static void skip_line(FILE *fp) {
    char c = fgetc(fp);

    while(c != '\n') {
        c = fgetc(fp);
    }
}


/*
 * parse_info: parses the info line "p cnf nbvar nbclauses"
 */
static void parse_info(
                       FILE *fp,
                       unsigned int *p_num_vars,
                       unsigned int *p_num_clauses
                      ) {
    char cnf[4];

    fscanf(fp, "%s %u %u", cnf, p_num_vars, p_num_clauses);

    char c = fgetc(fp);

    while(c != '\n') {
        c = fgetc(fp);
    }
}


/*
 * parse_clause: parses a clause
 */
static void parse_clause(
                         FILE *fp,
                         char c,
                         bool (*f_mat)[MAX_NUM_CLAUSES + 1][MAX_NUM_VARS * 2 + 1],
                         unsigned int *num_vars,
                         unsigned int *num_lits,
                         unsigned int *num_clauses,
                         unsigned int *p_num_vars
                        ) {
    (*num_clauses)++;

    char d = c;

    while(d != '0') {
        if('1' <= d && d <= '9') {
            // parse a positive literal
            parse_lit(fp, d, f_mat, num_vars, num_lits, num_clauses, p_num_vars);
        } else if(d == '-') {
            // parse a negative literal
            parse_lit(fp, '0', f_mat, num_vars, num_lits, num_clauses, p_num_vars);
        }

        d = fgetc(fp);
    }
}


/*
 * parse_lit: parses a lit
 */
static void parse_lit(
                      FILE *fp,
                      char d,
                      bool (*f_mat)[MAX_NUM_CLAUSES + 1][MAX_NUM_VARS * 2 + 1],
                      unsigned int *num_vars,
                      unsigned int *num_lits,
                      unsigned int *num_clauses,
                      unsigned int *p_num_vars
                     ) {
    int lit = d - '0';
    char c = fgetc(fp);
    
    while('0' <= c && c <= '9') {
        lit *= 10;
        lit += c - '0';
        c = fgetc(fp);
    }

    (*num_lits)++;
    *num_vars = MAX(*num_vars, lit);

    if(d == '0') {
        lit *= -1;
        unsigned int idx = LIT_TO_IDX(lit, *p_num_vars);
        (*f_mat)[*num_clauses][idx] = 1;
    } else {
        unsigned int idx = LIT_TO_IDX(lit, *p_num_vars);
        (*f_mat)[*num_clauses][idx] = 1;
    }
}


/*
 * init_formula: initializes the formula
 */
static void init_formula(
                         bool (*f_mat)[MAX_NUM_CLAUSES + 1][MAX_NUM_VARS * 2 + 1],
                         unsigned int num_vars,
                         unsigned int num_lits,
                         unsigned int num_clauses
                        ) {
    phi->dec_lvl = 0;
    init_csc(f_mat, num_vars, num_clauses);
    init_csr(f_mat, num_vars, num_clauses);
    init_vars(num_vars);
    init_clauses(num_clauses);
    init_lits(num_vars, num_lits);
}


/*
 * init_csc: initializes the CSC format
 */
static void init_csc(
                     bool (*f_mat)[MAX_NUM_CLAUSES + 1][MAX_NUM_VARS * 2 + 1],
                     unsigned int num_vars,
                     unsigned int num_clauses
                    ) {
    unsigned int num_lits = 0;
    
    unsigned int l;
    for(l = 1; l <= num_vars * 2; l++) {
        phi->csc_col_ptr[l] = num_lits + 1;

        for(unsigned int c = 1; c <= num_clauses; c++) {
            if((*f_mat)[c][l]) {
                num_lits++;
                phi->csc_row_ind[num_lits] = c;
            }
        }
    }

    phi->csc_col_ptr[l] = num_lits + 1;
}


/*
 * init_csr: initializes the CSR format
 */
static void init_csr(
                     bool (*f_mat)[MAX_NUM_CLAUSES + 1][MAX_NUM_VARS * 2 + 1],
                     unsigned int num_vars,
                     unsigned int num_clauses
                    ) {
    unsigned int num_lits = 0;

    unsigned int c;
    for(c = 1; c <= num_clauses; c++) {
        phi->csr_row_ptr[c] = num_lits + 1;

        for(unsigned int l = 1; l <= num_vars * 2; l++) {
            if((*f_mat)[c][l]) {
                num_lits++;
                phi->csr_col_ind[num_lits] = l;
            }
        }
    }

    phi->csr_row_ptr[c] = num_lits + 1;
}


/*
 * init_vars: initializes the variables data structures
 */
static void init_vars(unsigned int num_vars) {
    phi->s_num_vars = phi->d_num_vars = num_vars;

    for(unsigned int v = 1; v <= num_vars; v++) {
        phi->var_ass[v] = 0;
    }
}


/*
 * init_clauses: initializes the clauses data structures
 */
static void init_clauses(unsigned int num_clauses) {
    phi->s_num_clauses = phi->d_num_clauses = num_clauses;

    for(unsigned int c = 1; c <= num_clauses; c++) {
        phi->clause_sat[c] = 0;
        phi->s_clause_len[c] = phi->d_clause_len[c] = phi->csr_row_ptr[c+1] - phi->csr_row_ptr[c];
    }
}


/*
 * init_lits: initializes the literals data structures
 */
static void init_lits(unsigned int num_vars, unsigned int num_lits) {
    phi->s_num_lits = phi->d_num_lits = num_lits;

    for(unsigned int l = 1; l <= num_vars * 2; l++) {
        phi->s_occ_lits[l] = phi->d_occ_lits[l] = phi->csc_col_ptr[l+1] - phi->csc_col_ptr[l];
    }
}


/*
 * ass_lit_updates: updates the formula data structure based on the newly assigned
 *                  literals
 */
static void ass_lit_updates(int *new_lit_ass, unsigned int new_lit_ass_len) {
    for(unsigned int l = 0; l < new_lit_ass_len; l++) {
        int new_lit = new_lit_ass[l];
        
        // updates the satisfiability of clauses and the number of clauses NOT
        // yet satisfied
        unsigned int new_lit_idx = LIT_TO_IDX(new_lit, phi->s_num_vars);
        unsigned int clause_start_idx = phi->csc_col_ptr[new_lit_idx];

        for(unsigned int clause_off = 0; clause_off < phi->s_occ_lits[new_lit_idx]; clause_off++) {
            unsigned int clause_idx = phi->csc_row_ind[clause_start_idx + clause_off];

            if(!(phi->clause_sat[clause_idx])) {
                phi->clause_sat[clause_idx] = phi->dec_lvl;
                phi->d_num_clauses--;
                
                // updates the number of occurrences of literals and the number
                // of literals still active
                unsigned int lit_start_idx = phi->csr_row_ptr[clause_idx];

                for(unsigned int lit_off = 0; lit_off < phi->s_clause_len[clause_idx]; lit_off++) {
                    unsigned int lit_idx = phi->csr_col_ind[lit_start_idx + lit_off];
                    phi->d_occ_lits[lit_idx]--;

                    if(!phi->var_ass[IDX_TO_VAR(lit_idx, phi->s_num_vars)]) {
                        phi->d_num_lits--;
                    }
                }
            }
        }

        // updates the variable assignments and the number of variables NOT yet
        // assigned
        unsigned int var = LIT_TO_VAR(new_lit);

        if(new_lit > 0) {
            phi->var_ass[var] = phi->dec_lvl;
        } else {    // new_lit < 0
            phi->var_ass[var] = -(phi->dec_lvl);
        }

        phi->d_num_vars--;
    }
}


/*
 * neg_lit_updates: updates the formula data structure based on the negates of
 *                  the newly assigned literals
 */
static void neg_lit_updates(int *new_lit_ass, unsigned int new_lit_ass_len) {
    for(unsigned int l = 0; l < new_lit_ass_len; l++) {
        // updates the clause lengths and the number of literals still active
        int neg_lit = -new_lit_ass[l];
        unsigned int neg_lit_idx = LIT_TO_IDX(neg_lit, phi->s_num_vars);
        unsigned int clause_start_idx = phi->csc_col_ptr[neg_lit_idx];

        for(unsigned int clause_off = 0; clause_off < phi->s_occ_lits[neg_lit_idx]; clause_off++) {
            unsigned int clause_idx = phi->csc_row_ind[clause_start_idx + clause_off];

            if(!(phi->clause_sat[clause_idx])) {
                phi->d_clause_len[clause_idx]--;
                phi->d_num_lits--;
            }
        }
    }
}


/*
 * ass_lit_restores: restores the formula data structure based on the assigned
 *                   literals
 */
static void ass_lit_restores(int *lits_restore, unsigned int lits_restore_len, unsigned int bj_dec_lvl) {
    for(unsigned int l = 0; l < lits_restore_len; l++) {
        // restores the number of occurrences of literals and the number of
        // literals still active
        int res_lit = lits_restore[l];
        unsigned int res_lit_idx = LIT_TO_IDX(res_lit, phi->s_num_vars);
        unsigned int clause_start_idx = phi->csc_col_ptr[res_lit_idx];

        for(unsigned int clause_off = 0; clause_off < phi->s_occ_lits[res_lit_idx]; clause_off++) {
            unsigned int clause_idx = phi->csc_row_ind[clause_start_idx + clause_off];

            if(phi->clause_sat[clause_idx] > bj_dec_lvl) {
                unsigned int lit_start_idx = phi->csr_row_ptr[clause_idx];

                for(unsigned int lit_off = 0; lit_off < phi->s_clause_len[clause_idx]; lit_off++) {
                    unsigned int lit_idx = phi->csr_col_ind[lit_start_idx + lit_off];
                    phi->d_occ_lits[lit_idx]++;
                    phi->d_num_lits++;
                }

                // restores the satisfiability of clauses and the number of
                // clauses NOT yet satisfied
                phi->clause_sat[clause_idx] = 0;
                phi->d_num_clauses++;
            }
        }

        // restores the variable assignments and the number of variables NOT
        // yet assigned
        unsigned int var = LIT_TO_VAR(res_lit);
        phi->var_ass[var] = 0;
        phi->d_num_vars++;
    }
}


/*
 * neg_lit_restores: restores the formula data structure based on the negates
 *                   of the assigned literals
 */
static void neg_lit_restores(int *lits_restore, unsigned int lits_restore_len) {
    for(unsigned int l = 0; l < lits_restore_len; l++) {
        // restores the clause lengths and the number of literals still active
        int neg_lit = -lits_restore[l];
        unsigned int var = LIT_TO_VAR(neg_lit);
        unsigned int neg_lit_idx = LIT_TO_IDX(neg_lit, phi->s_num_vars);
        unsigned int clause_start_idx = phi->csc_col_ptr[neg_lit_idx];

        for(unsigned int clause_off = 0; clause_off < phi->s_occ_lits[neg_lit_idx]; clause_off++) {
            unsigned int clause_idx = phi->csc_row_ind[clause_start_idx + clause_off];

            if(!(phi->clause_sat[clause_idx]) || phi->clause_sat[clause_idx] > abs(phi->var_ass[var])) {
                phi->d_clause_len[clause_idx]++;
                phi->d_num_lits++;
            }
        }
    }
}


/*****************************************************************************/
