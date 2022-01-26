/**
 * miracle_dynamic.cuh: definition of the Miracle_Dyn data type and declaration
 * of its API.
 * 
 * Copyright (c) Michele Collevati
 */


#ifndef MIRACLE_DYNAMIC_CUH
#define MIRACLE_DYNAMIC_CUH


#include "cnf_formula.cuh"


/**
 * @brief Miracle_Dyn data type.
 */
typedef struct miracle_dyn {
    CNF_Formula *phi;           // Formula.

    int dec_lvl;                // Decision level starting from 1.

    /**
     * Variables
     */
    int *var_ass;               /** 
                                 * Array of variable assignments.
                                 *
                                 * var_ass[v] = 0
                                 * if variable v is unassigned.
                                 * 
                                 * var_ass[v] = dec_lvl
                                 * if variable v has been POSITIVELY assigned
                                 * at decision level dec_lvl.
                                 * 
                                 * var_ass[v] = -dec_lvl
                                 * if variable v has been NEGATIVELY assigned
                                 * at decision level dec_lvl.
                                 */
    int var_ass_len;            // Length of var_ass, which is phi->num_vars.
    int num_unass_vars;         // Number of unassigned variables.

    /**
     * Clauses
     */
    int *clause_sat;            /**
                                 * Array of clause satisfiability.
                                 *
                                 * clause_sat[c] = 0
                                 * if clause c is unresolved.
                                 * 
                                 * clause_sat[c] = dec_lvl
                                 * if clause c has been satisfied at decision
                                 * level dec_lvl.
                                 */
    int clause_sat_len;         /**
                                 * Length of clause_sat, which is
                                 * phi->num_clauses.
                                 */
    int num_unres_clauses;      // Number of unresolved clauses.
    int *clause_size;           // Array of clause initial sizes.
    int *unres_clause_size;     // Array of unresolved clause current sizes.
    int clause_size_len;        /**
                                 * Length of clause_size and unres_clause_size,
                                 * which is phi->num_clauses.
                                 */

    /**
     * Literals
     */
    int num_unres_lits;         // Number of unresolved literals.
    int *lit_occ;               // Array of literal initial occurrences.
    int *unres_lit_occ;         /**
                                 * Array of unresolved literal current
                                 * occurrences.
                                 */
    int lit_occ_len;            /**
                                 * Length of lit_occ and unres_lit_occ, which
                                 * is phi->num_vars * 2.
                                 */

    /**
     * Compressed Sparse Column Format (CSC)
     * Ref.: https://docs.nvidia.com/cuda/cusparse/index.html#csc-format
     */
    int *csc_row_ind;           // Array of clauses where literals occur.
    int csc_row_ind_len;        // Length of csc_row_ind, which is num_lits.
    int *csc_col_ptr;           /**
                                 * Array of literal index of csc_row_ind.
                                 *
                                 * The first n entries of this array contain
                                 * the indices to the first clause in the i-th
                                 * column for i = 0,...,n-1, while the last
                                 * entry contains num_lits + csc_col_ptr[0].
                                 * Since csc_col_ptr[0] = 0 for zero-based
                                 * indexing, num_lits + csc_col_ptr[0] =
                                 * num_lits.
                                 */
    int csc_col_ptr_len;        /**
                                 * Length of csc_col_ptr, which is
                                 * num_vars * 2 + 1.
                                 */
} Miracle_Dyn;


/**
 * API
 */


/**
 * @brief Creates a dynamic miracle.
 * 
 * @param [in]filename Filename of a formula in DIMACS CNF format.
 * @retval An initialized dynamic miracle.
 */
Miracle_Dyn *mrc_dyn_create_miracle(char *filename);


/**
 * @brief Destroys a dynamic miracle.
 * 
 * @param [in]mrc_dyn A dynamic miracle.
 * @retval None.
 */
void mrc_dyn_destroy_miracle(Miracle_Dyn *mrc_dyn);


/**
 * @brief Prints a dynamic miracle.
 * 
 * @param [in]mrc_dyn A dynamic miracle.
 * @retval None.
 */
void mrc_dyn_print_miracle(Miracle_Dyn *mrc_dyn);


/**
 * @brief Assigns literals in a dynamic miracle.
 * NOTE: To ensure the correctness of a Miracle_Dyn struct, the literal
 * assignments at each level must be done in one-shot with a SINGLE call to
 * mrc_dyn_assign_lits. This is due to how the struct and functions to update
 * it are designed.
 * 
 * @param [in]lits An array of assigned literals.
 * @param [in]lits_len Length of lits, which is the number of assigned
 * literals.
 * @param [in/out]mrc_dyn A dynamic miracle.
 * @retval None.
 */
void mrc_dyn_assign_lits(Lit *lits, int lits_len, Miracle_Dyn *mrc_dyn);


/**
 * @brief Increases the decision level in a dynamic miracle.
 * 
 * @param [in/out]mrc_dyn A dynamic miracle.
 * @retval None.
 */
inline void mrc_dyn_increase_decision_level(Miracle_Dyn *mrc_dyn) {
    mrc_dyn->dec_lvl++;
}


/**
 * @brief Backjumps to a decision level in a dynamic miracle.
 * 
 * @param [in]bj_dec_lvl A backjump decision level. A bj_dec_lvl < 1 resets the
 * dynamic miracle.
 * @param [in/out] A dynamic miracle.
 * @retval None.
 */
void mrc_dyn_backjump(int bj_dec_lvl, Miracle_Dyn *mrc_dyn);


/**
 * @brief Computes the RAND heuristic.
 * 
 * @param [in]mrc_dyn A dynamic miracle.
 * @retval The branching literal.
 */
Lit mrc_dyn_RAND_heuristic(Miracle_Dyn *mrc_dyn);


/**
 * @brief Computes the JW-OS heuristic.
 * 
 * @param [in]mrc_dyn A dynamic miracle.
 * @retval The branching literal.
 */
Lit mrc_dyn_JW_OS_heuristic(Miracle_Dyn *mrc_dyn);


/**
 * @brief Computes the JW-TS heuristic.
 * 
 * @param [in]mrc_dyn A dynamic miracle.
 * @retval The branching literal.
 */
Lit mrc_dyn_JW_TS_heuristic(Miracle_Dyn *mrc_dyn);


/**
 * @brief Computes the BOHM heuristic.
 *
 * @param [in]mrc_dyn A dynamic miracle.
 * @param [in]alpha A constant of the BOHM weight function.
 * @param [in]beta A constant of the BOHM weight function.
 * @retval The branching literal.
 */
Lit mrc_dyn_BOHM_heuristic(Miracle_Dyn *mrc_dyn,
                           const int alpha,
                           const int beta);


/**
 * @brief Computes the POSIT heuristic.
 * 
 * @param [in]mrc_dyn A dynamic miracle.
 * @param [in]n A constant of the POSIT weight function.
 * @retval The branching literal.
 */
Lit mrc_dyn_POSIT_heuristic(Miracle_Dyn *mrc_dyn, const int n);


/**
 * @brief Computes the DLIS heuristic.
 * 
 * @param [in]mrc_dyn A dynamic miracle.
 * @retval The branching literal.
 */
Lit mrc_dyn_DLIS_heuristic(Miracle_Dyn *mrc_dyn);


/**
 * @brief Computes the DLCS heuristic.
 * 
 * @param [in]mrc_dyn A dynamic miracle.
 * @retval The branching literal.
 */
Lit mrc_dyn_DLCS_heuristic(Miracle_Dyn *mrc_dyn);


/**
 * @brief Computes the RDLIS heuristic.
 * 
 * @param [in]mrc_dyn A dynamic miracle.
 * @retval The branching literal.
 */
Lit mrc_dyn_RDLIS_heuristic(Miracle_Dyn *mrc_dyn);


/**
 * @brief Computes the RDLCS heuristic.
 * 
 * @param [in]mrc_dyn A dynamic miracle.
 * @retval The branching literal.
 */
Lit mrc_dyn_RDLCS_heuristic(Miracle_Dyn *mrc_dyn);


#endif
