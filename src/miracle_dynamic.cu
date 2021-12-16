/**
 * miracle_dynamic.cu: definition of the Miracle_Dyn API.
 * 
 * Copyright (c) Michele Collevati
 */


#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <limits.h>


#include "miracle_dynamic.cuh"
#include "utils.cuh"


/**
 * @brief Clause data type.
 */
typedef struct clause {
    int size;       // Clause size.
    int idx;        // Clause index.
} Clause;


/**
 * Global variables
 */


Lidx *lidxs;                        /**
                                     * Array of assigned literal indices to
                                     * restore.
                                     */
int lidxs_len;                      /**
                                     * Length of lidxs, which is the number of
                                     * assigned literals to restore.
                                     */

static int *lit_occ;                // Array of literal occurrences.
static int lit_occ_len;             /**
                                     * Length of lit_occ, which is
                                     * mrc->phi->num_vars * 2.
                                     */

static Clause *clauses;             // Array of Clauses.
static int clauses_len;             /**
                                     * Length of clauses, which is
                                     * mrc->phi->num_clauses.
                                     */

static int *clause_indices;         // Array of Clause indices.
static int clause_indices_len;      /**
                                     * Current length of clause_indices,
                                     * which is the number of different sizes
                                     * + 1.
                                     */

static bool *var_availability;      // Array of variable availability.
static int var_availability_len;    /**
                                     * Length of var_availability, which is
                                     * mrc->phi->num_vars.
                                     */

static float *var_weights;          // Array of variable weights.
static int var_weights_len;         /**
                                     * Length of var_weights, which is
                                     * mrc->phi->num_vars.
                                     */


/**
 * Auxiliary function prototypes
 */


/**
 * @brief Initializes auxiliary data structures.
 * 
 * @param [in]mrc_dyn A dynamic miracle.
 * @retval None.
 */
static void init_aux_data_structs(Miracle_Dyn *mrc_dyn);


/**
 * @brief Destroys auxiliary data structures.
 * 
 * @retval None.
 */
static void destroy_aux_data_structs();


/**
 * @brief Updates due to assigned literals.
 * 
 * @param [in]lits An array of assigned literals.
 * @param [in]lits_len Length of lits, which is the number of assigned
 * literals.
 * @param [in/out]mrc_dyn A dynamic miracle.
 * @retval None.
 */
static void updates_assigned_lits(Lit *lits,
                                  int lits_len,
                                  Miracle_Dyn *mrc_dyn);


/**
 * @brief Updates due to negated assigned literals.
 * 
 * @param [in]lits An array of assigned literals.
 * @param [in]lits_len Length of lits, which is the number of assigned
 * literals.
 * @param [in/out]mrc_dyn A dynamic miracle.
 * @retval None.
 */
static void updates_neg_assigned_lits(Lit *lits,
                                      int lits_len,
                                      Miracle_Dyn *mrc_dyn);


/**
 * @brief Restores due to negated assigned literals.
 * 
 * @param [in/out]mrc_dyn A dynamic miracle.
 * @retval None.
 */
static void restores_neg_assigned_lits(Miracle_Dyn *mrc_dyn);


/**
 * @brief Restores due to assigned literals.
 * 
 * @param [in]bj_dec_lvl A backjump decision level. A bj_dec_lvl < 1 resets the
 * dynamic miracle.
 * @param [in/out]mrc_dyn A dynamic miracle.
 * @retval None.
 */
static void restores_assigned_lits(int bj_dec_lvl, Miracle_Dyn *mrc_dyn);


/**
 * @brief If two_sided = true, computes the JW-TS heuristic,
 * otherwise computes the JW-OS heuristic.
 * 
 * @param [in]mrc_dyn A dynamic miracle.
 * @param [in]two_sided A flag to choose JW-TS or JW-OS.
 * @retval The branching literal.
 */
static Lit JW_xS_heuristic(Miracle_Dyn *mrc_dyn, bool two_sided);


/**
 * @brief If dlcs = true, computes the DLCS heuristic,
 * otherwise computes the DLIS heuristic.
 * 
 * @param [in]mrc_dyn A dynamic miracle.
 * @param [in]dlcs A flag to choose DLCS or DLIS.
 * @retval The branching literal.
 */
static Lit DLxS_heuristic(Miracle_Dyn *mrc_dyn, bool dlcs);


/**
 * @brief If rdlcs = true, computes the RDLCS heuristic,
 * otherwise computes the RDLIS heuristic.
 * 
 * @param [in]mrc_dyn A dynamic miracle.
 * @param [in]rdlcs A flag to choose RDLCS or RDLIS.
 * @retval The branching literal.
 */
static Lit RDLxS_heuristic(Miracle_Dyn *mrc_dyn, bool rdlcs);


/**
 * @brief Compares two Clause elements.
 *
 * @param [in]a The first Clause element.
 * @param [in]b The second Clause element.
 * @retval A value that specifies the relationship between the two Clause
 * elements.
 */
static int compare_clauses(const void *a, const void *b);


/**
 * API definition
 */


Miracle_Dyn *mrc_dyn_create_miracle(char *filename) {
    Miracle_Dyn *mrc_dyn = (Miracle_Dyn *)malloc(sizeof *mrc_dyn);
    
    mrc_dyn->phi = cnf_parse_DIMACS(filename);

    mrc_dyn->dec_lvl = 1;
    
    /**
     * Variables.
     */
    mrc_dyn->var_ass_len = mrc_dyn->phi->num_vars;
    mrc_dyn->var_ass = (int *)calloc(mrc_dyn->var_ass_len,
                                     sizeof *(mrc_dyn->var_ass));

    mrc_dyn->num_unass_vars = mrc_dyn->phi->num_vars;

    /**
     * Clauses.
     */
    mrc_dyn->clause_sat_len = mrc_dyn->phi->num_clauses;
    mrc_dyn->clause_sat = (int *)calloc(mrc_dyn->clause_sat_len,
                                        sizeof *(mrc_dyn->clause_sat));

    mrc_dyn->num_unres_clauses = mrc_dyn->phi->num_clauses;

    mrc_dyn->clause_size_len = mrc_dyn->phi->num_clauses;
    mrc_dyn->clause_size = (int *)malloc(sizeof *(mrc_dyn->clause_size) *
                                         mrc_dyn->clause_size_len);
    mrc_dyn->unres_clause_size = (int *)malloc(
                                        sizeof *(mrc_dyn->unres_clause_size) *
                                        mrc_dyn->clause_size_len
                                              );

    int c_size;     // Clause size.

    for (int c = 0; c < mrc_dyn->phi->num_clauses; c++) {
        c_size = mrc_dyn->phi->clause_indices[c+1] -
                 mrc_dyn->phi->clause_indices[c];
        mrc_dyn->clause_size[c] = c_size;
        mrc_dyn->unres_clause_size[c] = c_size;
    }

    /**
     * Literals.
     */
    mrc_dyn->num_unres_lits = mrc_dyn->phi->num_lits;

    mrc_dyn->lit_occ_len = mrc_dyn->phi->num_vars * 2;
    mrc_dyn->lit_occ = (int *)calloc(mrc_dyn->lit_occ_len,
                                     sizeof *(mrc_dyn->lit_occ));
    mrc_dyn->unres_lit_occ = (int *)calloc(mrc_dyn->lit_occ_len,
                                           sizeof *(mrc_dyn->unres_lit_occ));

    Lidx lidx;

    for (int l = 0; l < mrc_dyn->phi->num_lits; l++) {
        lidx = mrc_dyn->phi->clauses[l];
        mrc_dyn->lit_occ[lidx]++;
        mrc_dyn->unres_lit_occ[lidx]++;
    }

    /**
     * Compressed Sparse Column Format (CSC)
     */
    mrc_dyn->csc_col_ptr_len = mrc_dyn->phi->num_vars * 2 + 1;
    mrc_dyn->csc_col_ptr = (int *)malloc(sizeof *(mrc_dyn->csc_col_ptr) *
                                         mrc_dyn->csc_col_ptr_len);

    mrc_dyn->csc_col_ptr[0] = 0;

    for (int l = 1; l < mrc_dyn->csc_col_ptr_len; l++) {
        mrc_dyn->csc_col_ptr[l] = mrc_dyn->csc_col_ptr[l-1] +
                                  mrc_dyn->lit_occ[l-1];
    }

    mrc_dyn->csc_row_ind_len = mrc_dyn->phi->num_lits;
    mrc_dyn->csc_row_ind = (int *)malloc(sizeof *(mrc_dyn->csc_row_ind) *
                                         mrc_dyn->csc_row_ind_len);

    int cidx;   // Index of csc_row_ind where to put the current clause.
    
    // Occurrences of processed literals.
    int *proc_lit_occ = (int *)calloc(mrc_dyn->phi->num_vars * 2,
                                      sizeof *proc_lit_occ);

    for (int c = 0; c < mrc_dyn->phi->num_clauses; c++) {
        for (int l = mrc_dyn->phi->clause_indices[c];
             l < mrc_dyn->phi->clause_indices[c+1];
             l++) {
            lidx = mrc_dyn->phi->clauses[l];
            cidx = mrc_dyn->csc_col_ptr[lidx] + proc_lit_occ[lidx];
            mrc_dyn->csc_row_ind[cidx] = c;
            proc_lit_occ[lidx]++;
        }
    }

    free(proc_lit_occ);

    init_aux_data_structs(mrc_dyn);

    return mrc_dyn;
}


void mrc_dyn_destroy_miracle(Miracle_Dyn *mrc_dyn) {
    cnf_destroy_formula(mrc_dyn->phi);
    free(mrc_dyn->var_ass);
    free(mrc_dyn->clause_sat);
    free(mrc_dyn->clause_size);
    free(mrc_dyn->unres_clause_size);
    free(mrc_dyn->lit_occ);
    free(mrc_dyn->unres_lit_occ);
    free(mrc_dyn->csc_row_ind);
    free(mrc_dyn->csc_col_ptr);
    free(mrc_dyn);
    destroy_aux_data_structs();
}


void mrc_dyn_print_miracle(Miracle_Dyn *mrc_dyn) {
    printf("*** Dynamic MiraCle ***\n\n");

    cnf_print_formula(mrc_dyn->phi);

    printf("Decision level: %d\n", mrc_dyn->dec_lvl);

    printf("Variable assignments: ");
    for (int v = 0; v < mrc_dyn->var_ass_len; v++) {
        printf("%d ", mrc_dyn->var_ass[v]);
    }
    printf("\n");

    printf("Number of unassigned variables: %d\n", mrc_dyn->num_unass_vars);

    printf("Clause satisfiability: ");
    for (int c = 0; c < mrc_dyn->clause_sat_len; c++) {
        printf("%d ", mrc_dyn->clause_sat[c]);
    }
    printf("\n");

    printf("Number of unresolved clauses: %d\n", mrc_dyn->num_unres_clauses);

    printf("Clause initial sizes: ");
    for (int c = 0; c < mrc_dyn->clause_size_len; c++) {
        printf("%d ", mrc_dyn->clause_size[c]);
    }
    printf("\n");

    printf("Unresolved clause current sizes: ");
    for (int c = 0; c < mrc_dyn->clause_size_len; c++) {
        printf("%d ", mrc_dyn->unres_clause_size[c]);
    }
    printf("\n");

    printf("Number of unresolved literals: %d\n", mrc_dyn->num_unres_lits);

    printf("Literal initial occurrences: ");
    for (int l = 0; l < mrc_dyn->lit_occ_len; l++) {
        printf("%d ", mrc_dyn->lit_occ[l]);
    }
    printf("\n");

    printf("Unresolved literal current occurrences: ");
    for (int l = 0; l < mrc_dyn->lit_occ_len; l++) {
        printf("%d ", mrc_dyn->unres_lit_occ[l]);
    }
    printf("\n");

    printf("\n*** Compressed Sparse Column Format (CSC) ***\n\n");

    printf("csc_col_ptr: ");
    for (int l = 0; l < mrc_dyn->csc_col_ptr_len; l++) {
        printf("%d ", mrc_dyn->csc_col_ptr[l]);
    }
    printf("\n");

    printf("csc_row_ind: ");
    for (int c = 0; c < mrc_dyn->csc_row_ind_len; c++) {
        printf("%d ", mrc_dyn->csc_row_ind[c]);
    }
    printf("\n");

    printf("\n*** End Compressed Sparse Column Format (CSC) ***\n\n");

    printf("*** End Dynamic MiraCle ***\n\n");
}


void mrc_dyn_assign_lits(Lit *lits, int lits_len, Miracle_Dyn *mrc_dyn) {
    // Updates due to assigned literals.
    updates_assigned_lits(lits, lits_len, mrc_dyn);

    // Updates due to negated assigned literals.
    updates_neg_assigned_lits(lits, lits_len, mrc_dyn);
}


void mrc_dyn_backjump(int bj_dec_lvl, Miracle_Dyn *mrc_dyn) {
    lidxs_len = 0;
    int v_ass;

    for (Var v = 0; v < mrc_dyn->phi->num_vars; v++) {
        v_ass = mrc_dyn->var_ass[v];

        if (v_ass && (abs(v_ass) > bj_dec_lvl)) {
            lidxs[lidxs_len] = v_ass > 0 ? varpol_to_lidx(v, true) :
                                           varpol_to_lidx(v, false);
            
            lidxs_len++;
        }
    }

    // Restores due to negated assigned literals.
    restores_neg_assigned_lits(mrc_dyn);

    // Restores due to assigned literals.
    restores_assigned_lits(bj_dec_lvl, mrc_dyn);

    // Restore decision level.
    mrc_dyn->dec_lvl = bj_dec_lvl < 1 ? 1 : bj_dec_lvl;
}


Lit mrc_dyn_RAND_heuristic(Miracle_Dyn *mrc_dyn) {
    init_PRNG();

    int random = rand() % mrc_dyn->num_unass_vars;
    Var bvar = UNDEF_VAR;

    for (Var v = 0; v < mrc_dyn->phi->num_vars; v++) {
        if (!(mrc_dyn->var_ass[v])) {
            // Variable Selection Heuristic.
            if (random == 0) {
                bvar = v;
                break;
            }

            random--;
        }
    }

    if (bvar == UNDEF_VAR) {
        fprintf(stderr, "Undefined variable \"bvar\" in function "
                "\"mrc_dyn_RAND_heuristic\".\n");
        exit(EXIT_FAILURE);
    }

    // Polarity Selection Heuristic.
    return rand() % 2 ? varpol_to_lit(bvar, false) : varpol_to_lit(bvar, true);
}


Lit mrc_dyn_JW_OS_heuristic(Miracle_Dyn *mrc_dyn) {
    return JW_xS_heuristic(mrc_dyn, false);
}


Lit mrc_dyn_JW_TS_heuristic(Miracle_Dyn *mrc_dyn) {
    return JW_xS_heuristic(mrc_dyn, true);
}


Lit mrc_dyn_BOHM_heuristic(Miracle_Dyn *mrc_dyn,
                           const int alpha,
                           const int beta) {
    // Init var_availability.
    for (Var v = 0; v < var_availability_len; v++) {
        var_availability[v] = !((bool)mrc_dyn->var_ass[v]);
    }

    int c_size;     // Clause size.

    // Init clauses.
    for (int c = 0; c < clauses_len; c++) {
        c_size = 0;

        if (!(mrc_dyn->clause_sat[c])) {
            c_size = mrc_dyn->unres_clause_size[c];
        }

        clauses[c].size = c_size;
        clauses[c].idx = c;
    }

    // Sort the clauses by increasing size.
    qsort(clauses, clauses_len, sizeof *clauses, compare_clauses);

    // Build the array of Clause indices.
    clause_indices_len = 0;
    clause_indices[clause_indices_len] = 0;
    clause_indices_len++;
    
    for (int c = 1; c < clauses_len; c++) {
        if (clauses[c-1].size < clauses[c].size) {
            clause_indices[clause_indices_len] = c;
            clause_indices_len++;
        }
    }

    clause_indices[clause_indices_len] = clauses_len;
    clause_indices_len++;
    
    int c;
    float greatest_weight;
    Lidx lidx;
    Var var;
    Lidx pos_lidx;
    Lidx neg_lidx;
    int lc_i_pos_lidx;
    int lc_i_neg_lidx;
    float weight;
    Var bvar = UNDEF_VAR;

    for (int i = clauses[0].size == 0 ? 1 : 0;
         i < clause_indices_len - 1;
         i++) {
        // Clear lit_occ.
        memset(lit_occ, 0, sizeof *lit_occ * lit_occ_len);

        // Clear var_weights.
        memset(var_weights, 0, sizeof *var_weights * var_weights_len);

        // Reset greatest_weight.
        greatest_weight = -1.0;

        for (int cidx = clause_indices[i];
             cidx < clause_indices[i+1];
             cidx++) {
            c = clauses[cidx].idx;

            for (int l = mrc_dyn->phi->clause_indices[c];
                 l < mrc_dyn->phi->clause_indices[c+1];
                 l++) {
                lidx = mrc_dyn->phi->clauses[l];
                var = lidx_to_var(lidx);

                if (var_availability[var]) {
                    // Update lc_i(l).
                    lit_occ[lidx]++;

                    // Compute w_i(v).
                    pos_lidx = varpol_to_lidx(var, true);
                    neg_lidx = varpol_to_lidx(var, false);
                    lc_i_pos_lidx = lit_occ[pos_lidx];
                    lc_i_neg_lidx = lit_occ[neg_lidx];
                    weight = (float)
                             (alpha * max(lc_i_pos_lidx, lc_i_neg_lidx) +
                              beta * min(lc_i_pos_lidx, lc_i_neg_lidx));
                    var_weights[var] = weight;

                    // Compute the greatest w_i(v).
                    if (weight > greatest_weight) {
                        greatest_weight = weight;
                    }
                }
            }
        }

        // Variable Selection Heuristic.
        bvar = UNDEF_VAR;

        for (Var v = 0; v < var_availability_len; v++) {
            if (var_availability[v]) {
                if (var_weights[v] < greatest_weight) {
                    var_availability[v] = false;
                } else if (bvar == UNDEF_VAR) {
                    bvar = v;
                }
            }
        }
    }

    if (bvar == UNDEF_VAR) {
        fprintf(stderr, "Undefined variable \"bvar\" in function "
                "\"mrc_dyn_BOHM_heuristic\".\n");
        exit(EXIT_FAILURE);
    }

    pos_lidx = varpol_to_lidx(bvar, true);
    neg_lidx = varpol_to_lidx(bvar, false);
    lc_i_pos_lidx = mrc_dyn->unres_lit_occ[pos_lidx];
    lc_i_neg_lidx = mrc_dyn->unres_lit_occ[neg_lidx];

    // Polarity Selection Heuristic.
    return lc_i_pos_lidx >= lc_i_neg_lidx ? lidx_to_lit(pos_lidx) :
                                            lidx_to_lit(neg_lidx);
}


Lit mrc_dyn_POSIT_heuristic(Miracle_Dyn *mrc_dyn, const int n) {
    // Compute the smallest clause size.
    int c_size;     // Clause size.
    int smallest_c_size = INT_MAX;      // Smallest clause size.

    for (int c = 0; c < mrc_dyn->clause_size_len; c++) {
        if (!(mrc_dyn->clause_sat[c])) {
            c_size = mrc_dyn->unres_clause_size[c];

            if (c_size < smallest_c_size) {
                smallest_c_size = c_size;
            }
        }
    }

    Lidx pos_lidx;
    Lidx neg_lidx;
    int clause;
    int lc_min_pos_lidx = 0;
    int lc_min_neg_lidx = 0;
    int weight;
    int greatest_weight = -1;
    Var bvar = UNDEF_VAR;
    int bvar_lc_min_pos_lidx;
    int bvar_lc_min_neg_lidx;

    for (Var v = 0; v < mrc_dyn->phi->num_vars; v++) {
        if (!(mrc_dyn->var_ass[v])) {
            // Variable Selection Heuristic.
            pos_lidx = varpol_to_lidx(v, true);
            neg_lidx = varpol_to_lidx(v, false);

            /**
             * Compute the number of occurrences of the positive literal in the
             * smallest unresolved clauses.
             */
            for (int c = mrc_dyn->csc_col_ptr[pos_lidx];
                 c < mrc_dyn->csc_col_ptr[pos_lidx+1];
                 c++) {
                clause = mrc_dyn->csc_row_ind[c];

                if (!(mrc_dyn->clause_sat[clause]) &&
                    mrc_dyn->unres_clause_size[clause] == smallest_c_size) {
                    lc_min_pos_lidx++;
                }
            }

            /**
             * Compute the number of occurrences of the negative literal in the
             * smallest unresolved clauses.
             */
            for (int c = mrc_dyn->csc_col_ptr[neg_lidx];
                 c < mrc_dyn->csc_col_ptr[neg_lidx+1];
                 c++) {
                clause = mrc_dyn->csc_row_ind[c];

                if (!(mrc_dyn->clause_sat[clause]) &&
                    mrc_dyn->unres_clause_size[clause] == smallest_c_size) {
                    lc_min_neg_lidx++;
                }
            }

            weight = lc_min_pos_lidx * lc_min_neg_lidx *
                     (int)(pow(2, n) + 0.5) +
                     lc_min_pos_lidx + lc_min_neg_lidx;
            
            if (weight > greatest_weight) {
                bvar = v;
                bvar_lc_min_pos_lidx = lc_min_pos_lidx;
                bvar_lc_min_neg_lidx = lc_min_neg_lidx;
                greatest_weight = weight;
            }

            lc_min_pos_lidx = 0;
            lc_min_neg_lidx = 0;
        }
    }

    if (bvar == UNDEF_VAR) {
        fprintf(stderr, "Undefined variable \"bvar\" in function "
                "\"mrc_dyn_POSIT_heuristic\".\n");
        exit(EXIT_FAILURE);
    }

    // Polarity Selection Heuristic.
    return bvar_lc_min_pos_lidx >= bvar_lc_min_neg_lidx ?
           varpol_to_lit(bvar, false) : varpol_to_lit(bvar, true);
}


Lit mrc_dyn_DLIS_heuristic(Miracle_Dyn *mrc_dyn) {
    return DLxS_heuristic(mrc_dyn, false);
}


Lit mrc_dyn_DLCS_heuristic(Miracle_Dyn *mrc_dyn) {
    return DLxS_heuristic(mrc_dyn, true);
}


Lit mrc_dyn_RDLIS_heuristic(Miracle_Dyn *mrc_dyn) {
    return RDLxS_heuristic(mrc_dyn, false);
}


Lit mrc_dyn_RDLCS_heuristic(Miracle_Dyn *mrc_dyn) {
    return RDLxS_heuristic(mrc_dyn, true);
}


/**
 * Auxiliary function definitions
 */


static void init_aux_data_structs(Miracle_Dyn *mrc_dyn) {
    lidxs_len = 0;
    lidxs = (Lidx *)malloc(sizeof *lidxs * mrc_dyn->phi->num_vars);

    lit_occ_len = mrc_dyn->phi->num_vars * 2;
    lit_occ = (int *)calloc(lit_occ_len,
                            sizeof *lit_occ);

    clauses_len = mrc_dyn->phi->num_clauses;
    clauses = (Clause *)malloc(sizeof *clauses * clauses_len);

    clause_indices_len = 0;
    clause_indices = (int *)malloc(sizeof *clause_indices *
                                   (clauses_len + 1));

    var_availability_len = mrc_dyn->phi->num_vars;
    var_availability = (bool *)malloc(sizeof *var_availability *
                                      var_availability_len);

    var_weights_len = mrc_dyn->phi->num_vars;
    var_weights = (float *)calloc(var_weights_len,
                                  sizeof *var_weights);
}


static void destroy_aux_data_structs() {
    free(lidxs);
    free(lit_occ);
    free(clauses);
    free(clause_indices);
    free(var_availability);
    free(var_weights);
}


static void updates_assigned_lits(Lit *lits,
                                  int lits_len,
                                  Miracle_Dyn *mrc_dyn) {
    Lit lit;
    Lidx lidx;
    int clause;
    Lidx clidx;     // Current clause literal index.
    Var var;
    bool pol;
    
    for (int lt = 0; lt < lits_len; lt++) {
        lit = lits[lt];
        lidx = lit_to_lidx(lit);

        for (int c = mrc_dyn->csc_col_ptr[lidx];
             c < mrc_dyn->csc_col_ptr[lidx+1];
             c++) {
            clause = mrc_dyn->csc_row_ind[c];

            if (!(mrc_dyn->clause_sat[clause])) {
                for (int l = mrc_dyn->phi->clause_indices[clause];
                     l < mrc_dyn->phi->clause_indices[clause+1];
                     l++) {
                    clidx = mrc_dyn->phi->clauses[l];
                    var = lidx_to_var(clidx);

                    if (!(mrc_dyn->var_ass[var]) ||
                        abs(mrc_dyn->var_ass[var]) == mrc_dyn->dec_lvl) {
                        // Update the unresolved literal current occurrences.
                        mrc_dyn->unres_lit_occ[clidx]--;
                        
                        // Update the number of unresolved literals.
                        mrc_dyn->num_unres_lits--;
                    }
                }

                // Update the clause satisfiability.
                mrc_dyn->clause_sat[clause] = mrc_dyn->dec_lvl;
                
                // Update the number of unresolved clauses.
                mrc_dyn->num_unres_clauses--;
            }
        }

        var = lit_to_var(lit);
        pol = lit_to_pol(lit);

        // Update the variable assignments.
        mrc_dyn->var_ass[var] = pol ? mrc_dyn->dec_lvl : -(mrc_dyn->dec_lvl);
    }

    // Update the number of unassigned variables.
    mrc_dyn->num_unass_vars -= lits_len;
}


static void updates_neg_assigned_lits(Lit *lits,
                                      int lits_len,
                                      Miracle_Dyn *mrc_dyn) {
    Lit lit;
    Lidx lidx;
    int clause;

    for (int lt = 0; lt < lits_len; lt++) {
        lit = neg_lit(lits[lt]);
        lidx = lit_to_lidx(lit);

        for (int c = mrc_dyn->csc_col_ptr[lidx];
             c < mrc_dyn->csc_col_ptr[lidx+1];
             c++) {
            clause = mrc_dyn->csc_row_ind[c];

            if (!(mrc_dyn->clause_sat[clause])) {
                // Update the unresolved clause current sizes.
                mrc_dyn->unres_clause_size[clause]--;
                
                // Update the unresolved literal current occurrences.
                mrc_dyn->unres_lit_occ[lidx]--;
                
                // Update the number of unresolved literals.
                mrc_dyn->num_unres_lits--;
            }
        }
    }
}


static void restores_neg_assigned_lits(Miracle_Dyn *mrc_dyn) {
    Lidx lidx;
    Var var;
    int clause;

    for (int lt = 0; lt < lidxs_len; lt++) {
        lidx = neg_lidx(lidxs[lt]);
        var = lidx_to_var(lidx);

        for (int c = mrc_dyn->csc_col_ptr[lidx];
             c < mrc_dyn->csc_col_ptr[lidx+1];
             c++) {
            clause = mrc_dyn->csc_row_ind[c];

            if (!(mrc_dyn->clause_sat[clause]) ||
                (mrc_dyn->clause_sat[clause] > abs(mrc_dyn->var_ass[var]))) {
                // Restore the unresolved clause current sizes.
                mrc_dyn->unres_clause_size[clause]++;

                // Restore the unresolved literal current occurrences.
                mrc_dyn->unres_lit_occ[lidx]++;

                // Restore the number of unresolved literals.
                mrc_dyn->num_unres_lits++;
            }
        }
    }
}


static void restores_assigned_lits(int bj_dec_lvl, Miracle_Dyn *mrc_dyn) {
    Lidx lidx;
    int clause;
    Lidx clidx;
    Var var;
    
    for (int lt = 0; lt < lidxs_len; lt++) {
        lidx = lidxs[lt];

        for (int c = mrc_dyn->csc_col_ptr[lidx];
             c < mrc_dyn->csc_col_ptr[lidx+1];
             c++) {
            clause = mrc_dyn->csc_row_ind[c];

            if (mrc_dyn->clause_sat[clause] > bj_dec_lvl) {
                for (int l = mrc_dyn->phi->clause_indices[clause];
                     l < mrc_dyn->phi->clause_indices[clause+1];
                     l++) {
                    clidx = mrc_dyn->phi->clauses[l];
                    var = lidx_to_var(clidx);

                    if (!(mrc_dyn->var_ass[var]) ||
                        (mrc_dyn->clause_sat[clause] <=
                        abs(mrc_dyn->var_ass[var]))) {
                        // Restore the unresolved literal current occurrences.
                        mrc_dyn->unres_lit_occ[clidx]++;

                        // Restore the number of unresolved literals.
                        mrc_dyn->num_unres_lits++;
                    }
                }

                // Restore the clause satisfiability.
                mrc_dyn->clause_sat[clause] = 0;

                // Restore the number of unresolved clauses.
                mrc_dyn->num_unres_clauses++;
            }
        }

        var = lidx_to_var(lidx);

        // Restore the variable assignments.
        mrc_dyn->var_ass[var] = 0;
    }

    // Restore the number of unassigned variables.
    mrc_dyn->num_unass_vars += lidxs_len;
}


static Lit JW_xS_heuristic(Miracle_Dyn *mrc_dyn, bool two_sided) {
    Lidx pos_lidx;
    Lidx neg_lidx;
    int clause;
    int c_size;     // Clause size.
    float weight_pos_lidx = 0;
    float weight_neg_lidx = 0;
    float weight;
    float greatest_weight = -1.0;
    Var bvar = UNDEF_VAR;
    float bvar_weight_pos_lidx;
    float bvar_weight_neg_lidx;
    
    for (Var v = 0; v < mrc_dyn->phi->num_vars; v++) {
        if (!(mrc_dyn->var_ass[v])) {
            // Variable Selection Heuristic.
            pos_lidx = varpol_to_lidx(v, true);
            neg_lidx = varpol_to_lidx(v, false);

            // Compute the positive literal weight.
            for (int c = mrc_dyn->csc_col_ptr[pos_lidx];
                 c < mrc_dyn->csc_col_ptr[pos_lidx+1];
                 c++) {
                clause = mrc_dyn->csc_row_ind[c];

                if (!(mrc_dyn->clause_sat[clause])) {
                    c_size = mrc_dyn->unres_clause_size[clause];
                    weight_pos_lidx += powf(2.0, (float)-c_size);
                }
            }

            // Compute the negative literal weight.
            for (int c = mrc_dyn->csc_col_ptr[neg_lidx];
                 c < mrc_dyn->csc_col_ptr[neg_lidx+1];
                 c++) {
                clause = mrc_dyn->csc_row_ind[c];

                if (!(mrc_dyn->clause_sat[clause])) {
                    c_size = mrc_dyn->unres_clause_size[clause];
                    weight_neg_lidx += powf(2.0, (float)-c_size);
                }
            }

            weight = two_sided ? abs(weight_pos_lidx - weight_neg_lidx) :
                                 (weight_pos_lidx >= weight_neg_lidx ?
                                  weight_pos_lidx : weight_neg_lidx);

            if (weight > greatest_weight) {
                bvar = v;
                bvar_weight_pos_lidx = weight_pos_lidx;
                bvar_weight_neg_lidx = weight_neg_lidx;
                greatest_weight = weight;
            }

            weight_pos_lidx = 0;
            weight_neg_lidx = 0;
        }
    }

    if (bvar == UNDEF_VAR) {
        fprintf(stderr, "Undefined variable \"bvar\" in function "
                "\"JW_xS_heuristic\".\n");
        exit(EXIT_FAILURE);
    }

    // Polarity Selection Heuristic.
    return bvar_weight_pos_lidx >= bvar_weight_neg_lidx ?
           varpol_to_lit(bvar, true) : varpol_to_lit(bvar, false);
}


static Lit DLxS_heuristic(Miracle_Dyn *mrc_dyn, bool dlcs) {
    Lidx pos_lidx;
    Lidx neg_lidx;
    int ulo_pos_lidx;
    int ulo_neg_lidx;
    int sum;
    int largest_sum = -1;
    Var bvar = UNDEF_VAR;
    int bvar_ulo_pos_lidx;
    int bvar_ulo_neg_lidx;

    for (Var v = 0; v < mrc_dyn->phi->num_vars; v++) {
        if (!(mrc_dyn->var_ass[v])) {
            // Variable Selection Heuristic.
            pos_lidx = varpol_to_lidx(v, true);
            neg_lidx = varpol_to_lidx(v, false);
            ulo_pos_lidx = mrc_dyn->unres_lit_occ[pos_lidx];
            ulo_neg_lidx = mrc_dyn->unres_lit_occ[neg_lidx];

            sum = dlcs ? ulo_pos_lidx + ulo_neg_lidx :
                         (ulo_pos_lidx >= ulo_neg_lidx ? ulo_pos_lidx :
                                                         ulo_neg_lidx);

            if (sum > largest_sum) {
                bvar = v;
                bvar_ulo_pos_lidx = ulo_pos_lidx;
                bvar_ulo_neg_lidx = ulo_neg_lidx;
                largest_sum = sum;
            }
        }
    }

    if (bvar == UNDEF_VAR) {
        fprintf(stderr, "Undefined variable \"bvar\" in function "
                "\"DLxS_heuristic\".\n");
        exit(EXIT_FAILURE);
    }

    // Polarity Selection Heuristic.
    return bvar_ulo_pos_lidx >= bvar_ulo_neg_lidx ? varpol_to_lit(bvar, true) :
                                                    varpol_to_lit(bvar, false);
}


static Lit RDLxS_heuristic(Miracle_Dyn *mrc_dyn, bool rdlcs) {
    init_PRNG();

    if (rdlcs && (rand() % 2)) {
        return neg_lit(mrc_dyn_DLCS_heuristic(mrc_dyn));
    } else if (rdlcs) {
        return mrc_dyn_DLCS_heuristic(mrc_dyn);
    } else if (rand() % 2) {
        return neg_lit(mrc_dyn_DLIS_heuristic(mrc_dyn));
    } else {
        return mrc_dyn_DLIS_heuristic(mrc_dyn);
    }
}


static int compare_clauses(const void *a, const void *b) {
    return (((Clause *)a)->size - ((Clause *)b)->size);
}
