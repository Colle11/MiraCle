/*
 * formula.cu
 *
 * file containing the formula library
 *
 */


#include <stdio.h>
#include <stdlib.h>


#include "formula.h"
#include "utilities.h"


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


// static limits to formula size
#define MAX_NUM_CLAUSES 550000      // maximum number of clauses
#define MAX_NUM_LITS    8000000     // maximum number of literals
#define MAX_NUM_VARS    1000        // maximum number of variables


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
 * external variables
 */

static formula *f;


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
    printf("decision level (dynamic): %u\n\n\n", f->dec_lvl);


    /*
     * variables
     */
    printf("*************\n");
    printf("* variables *\n");
    printf("*************\n");
    printf("number of variables (static): %u\n\n", f->s_num_vars);
    printf("number of variables NOT yet assigned (dynamic): %u\n\n", f->d_num_vars);
    printf("variable assignments (dynamic): ");
    for(unsigned int v = 1; v <= f->s_num_vars; v++) {
        printf("%i ", f->var_ass[v]);
    }
    printf("\n\n\n");


    /*
     * clauses
     */
    printf("***********\n");
    printf("* clauses *\n");
    printf("***********\n");
    printf("number of clauses (static): %u\n\n", f->s_num_clauses);
    printf("number of clauses NOT yet satisfied (dynamic): %u\n\n", f->d_num_clauses);
    printf("satisfiability of clauses (dynamic): ");
    for(unsigned int c = 1; c <= f->s_num_clauses; c++) {
        printf("%u ", f->clause_sat[c]);
    }
    printf("\n\n");
    printf("clause lengths (static): ");
    for(unsigned int c = 1; c <= f->s_num_clauses; c++) {
        printf("%u ", f->s_clause_len[c]);
    }
    printf("\n\n");
    printf("clause lengths (dynamic): ");
    for(unsigned int c = 1; c <= f->s_num_clauses; c++) {
        printf("%u ", f->d_clause_len[c]);
    }
    printf("\n\n\n");


    /*
     * literals
     */
    printf("************\n");
    printf("* literals *\n");
    printf("************\n");
    printf("number of literals (static): %u\n\n", f->s_num_lits);
    printf("number of literals still active (dynamic): %u\n\n", f->d_num_lits);
    printf("number of occurrences of literals (static): ");
    for(unsigned int l = 1; l <= f->s_num_vars * 2; l++) {
        printf("%u ", f->s_occ_lits[l]);
    }
    printf("\n\n");
    printf("number of occurrences of literals (dynamic): ");
    for(unsigned int l = 1; l <= f->s_num_vars * 2; l++) {
        printf("%u ", f->d_occ_lits[l]);
    }
    printf("\n\n\n");


    /*
     * Compressed Sparse Column Format (CSC)
     */
    printf("*****************************************\n");
    printf("* Compressed Sparse Column Format (CSC) *\n");
    printf("*****************************************\n");
    printf("csc_row_ind: ");
    for(unsigned int l = 1; l <= f->s_num_lits; l++) {
        printf("%u ", f->csc_row_ind[l]);
    }
    printf("\n\n");
    printf("csc_col_ptr: ");
    for(unsigned int l = 1; l <= f->s_num_vars * 2 + 1; l++) {
        printf("%u ", f->csc_col_ptr[l]);
    }
    printf("\n\n\n");

    
    /*
     * Compressed Sparse Row Format (CSR)
     */
    printf("**************************************\n");
    printf("* Compressed Sparse Row Format (CSR) *\n");
    printf("**************************************\n");
    printf("csr_col_ind: ");
    for(unsigned int l = 1; l <= f->s_num_lits; l++) {
        printf("%u ", f->csr_col_ind[l]);
    }
    printf("\n\n");
    printf("csr_row_ptr: ");
    for(unsigned int c = 1; c <= f->s_num_clauses + 1; c++) {
        printf("%u ", f->csr_row_ptr[c]);
    }
    printf("\n");
}


/*
 * new_lit_assignments: updates the formula data structure with the new literal
 *                      assignments
 */
void new_lit_assignments(int *new_lit_ass, unsigned int new_lit_ass_len) {
    // updates the decision level
    f->dec_lvl++;

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

    for(unsigned int v = 1; v <= f->s_num_vars; v++) {
        int v_ass = f->var_ass[v];
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
    f->dec_lvl = bj_dec_lvl;
}


/*****************************************************************************/


/*
 * auxiliary function definitions
 */

/*
 * alloc_formula: allocates the empty formula on the host memory
 */
static void alloc_formula() {
    HANDLE_ERROR( cudaHostAlloc( (void**) &f,
                                 sizeof( *f ),
                                 cudaHostAllocDefault ) );
}


/*
 * free_formula: deallocates the formula on the host memory
 */
static void free_formula() {
    HANDLE_ERROR( cudaFreeHost( f ) );
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
    f->dec_lvl = 0;
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
        f->csc_col_ptr[l] = num_lits + 1;

        for(unsigned int c = 1; c <= num_clauses; c++) {
            if((*f_mat)[c][l]) {
                num_lits++;
                f->csc_row_ind[num_lits] = c;
            }
        }
    }

    f->csc_col_ptr[l] = num_lits + 1;
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
        f->csr_row_ptr[c] = num_lits + 1;

        for(unsigned int l = 1; l <= num_vars * 2; l++) {
            if((*f_mat)[c][l]) {
                num_lits++;
                f->csr_col_ind[num_lits] = l;
            }
        }
    }

    f->csr_row_ptr[c] = num_lits + 1;
}


/*
 * init_vars: initializes the variables data structures
 */
static void init_vars(unsigned int num_vars) {
    f->s_num_vars = f->d_num_vars = num_vars;

    for(unsigned int v = 1; v <= num_vars; v++) {
        f->var_ass[v] = 0;
    }
}


/*
 * init_clauses: initializes the clauses data structures
 */
static void init_clauses(unsigned int num_clauses) {
    f->s_num_clauses = f->d_num_clauses = num_clauses;

    for(unsigned int c = 1; c <= num_clauses; c++) {
        f->clause_sat[c] = 0;
        f->s_clause_len[c] = f->d_clause_len[c] = f->csr_row_ptr[c+1] - f->csr_row_ptr[c];
    }
}


/*
 * init_lits: initializes the literals data structures
 */
static void init_lits(unsigned int num_vars, unsigned int num_lits) {
    f->s_num_lits = f->d_num_lits = num_lits;

    for(unsigned int l = 1; l <= num_vars * 2; l++) {
        f->s_occ_lits[l] = f->d_occ_lits[l] = f->csc_col_ptr[l+1] - f->csc_col_ptr[l];
    }
}


/*
 * ass_lit_updates: updates the formula data structure based on the newly assigned
 *                  literals
 */
static void ass_lit_updates(int *new_lit_ass, unsigned int new_lit_ass_len) {
    for(unsigned int l = 0; l < new_lit_ass_len; l++) {
        // updates the variable assignments and the number of variables NOT yet
        // assigned
        int new_lit = new_lit_ass[l];
        unsigned int var = LIT_TO_VAR(new_lit);

        if(new_lit > 0) {
            f->var_ass[var] = f->dec_lvl;
        } else {    // new_lit < 0
            f->var_ass[var] = -(f->dec_lvl);
        }

        f->d_num_vars--;

        // updates the satisfiability of clauses and the number of clauses NOT
        // yet satisfied
        unsigned int new_lit_idx = LIT_TO_IDX(new_lit, f->s_num_vars);
        unsigned int clause_start_idx = f->csc_col_ptr[new_lit_idx];

        for(unsigned int clause_off = 0; clause_off < f->s_occ_lits[new_lit_idx]; clause_off++) {
            unsigned int clause_idx = f->csc_row_ind[clause_start_idx + clause_off];

            if(!(f->clause_sat[clause_idx])) {
                f->clause_sat[clause_idx] = f->dec_lvl;
                f->d_num_clauses--;
                
                // updates the number of occurrences of literals and the number
                // of literals still active
                unsigned int lit_start_idx = f->csr_row_ptr[clause_idx];

                for(unsigned int lit_off = 0; lit_off < f->s_clause_len[clause_idx]; lit_off++) {
                    unsigned int lit_idx = f->csr_col_ind[lit_start_idx + lit_off];
                    f->d_occ_lits[lit_idx]--;
                    f->d_num_lits--;
                }
            }
        }
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
        unsigned int neg_lit_idx = LIT_TO_IDX(neg_lit, f->s_num_vars);
        unsigned int clause_start_idx = f->csc_col_ptr[neg_lit_idx];

        for(unsigned int clause_off = 0; clause_off < f->s_occ_lits[neg_lit_idx]; clause_off++) {
            unsigned int clause_idx = f->csc_row_ind[clause_start_idx + clause_off];

            if(!(f->clause_sat[clause_idx])) {
                f->d_clause_len[clause_idx]--;
                f->d_num_lits--;
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
        unsigned int res_lit_idx = LIT_TO_IDX(res_lit, f->s_num_vars);
        unsigned int clause_start_idx = f->csc_col_ptr[res_lit_idx];

        for(unsigned int clause_off = 0; clause_off < f->s_occ_lits[res_lit_idx]; clause_off++) {
            unsigned int clause_idx = f->csc_row_ind[clause_start_idx + clause_off];

            if(f->clause_sat[clause_idx] > bj_dec_lvl) {
                unsigned int lit_start_idx = f->csr_row_ptr[clause_idx];

                for(unsigned int lit_off = 0; lit_off < f->s_clause_len[clause_idx]; lit_off++) {
                    unsigned int lit_idx = f->csr_col_ind[lit_start_idx + lit_off];
                    f->d_occ_lits[lit_idx]++;
                    f->d_num_lits++;
                }

                // restores the satisfiability of clauses and the number of
                // clauses NOT yet satisfied
                f->clause_sat[clause_idx] = 0;
                f->d_num_clauses++;
            }
        }

        // restores the variable assignments and the number of variables NOT
        // yet assigned
        unsigned int var = LIT_TO_VAR(res_lit);
        f->var_ass[var] = 0;
        f->d_num_vars++;
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
        unsigned int neg_lit_idx = LIT_TO_IDX(neg_lit, f->s_num_vars);
        unsigned int clause_start_idx = f->csc_col_ptr[neg_lit_idx];

        for(unsigned int clause_off = 0; clause_off < f->s_occ_lits[neg_lit_idx]; clause_off++) {
            unsigned int clause_idx = f->csc_row_ind[clause_start_idx + clause_off];

            if(!(f->clause_sat[clause_idx]) || f->clause_sat[clause_idx] > abs(f->var_ass[var])) {
                f->d_clause_len[clause_idx]++;
                f->d_num_lits++;
            }
        }
    }
}


/*****************************************************************************/
