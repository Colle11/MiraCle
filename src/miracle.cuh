/**
 * miracle.cuh: definition of the Miracle data type and declaration of its API.
 * 
 * Copyright (c) Michele Collevati
 */


#ifndef MIRACLE_CUH
#define MIRACLE_CUH


#include "cnf_formula.cuh"


struct sat_miracle;


/**
 * @brief Miracle data type.
 */
typedef struct miracle {
    CNF_Formula *phi;   // Formula.

    int dec_lvl;        // Decision level starting from 1.
    
    int *var_ass;       /** 
                         * Array of variable assignments.
                         *
                         * var_ass[v] = 0
                         * if variable v is unassigned.
                         * 
                         * var_ass[v] = dec_lvl
                         * if variable v has been POSITIVELY assigned at
                         * decision level dec_lvl.
                         * 
                         * var_ass[v] = -dec_lvl
                         * if variable v has been NEGATIVELY assigned at
                         * decision level dec_lvl.
                         */
    int var_ass_len;    // Length of var_ass, which is phi->num_vars.
    
    int *clause_sat;    /**
                         * Array of clause satisfiability.
                         *
                         * clause_sat[c] = 0
                         * if clause c is unresolved.
                         * 
                         * clause_sat[c] = dec_lvl
                         * if clause c has been satisfied at decision level
                         * dec_lvl.
                         */
    int clause_sat_len; // Length of clause_sat, which is phi->num_clauses.
} Miracle;


/**
 * API
 */


/**
 * @brief Creates a miracle.
 * 
 * @param [in]filename Filename of a formula in DIMACS CNF format.
 * @retval An initialized miracle.
 */
Miracle *mrc_create_miracle(char *filename);


/**
 * @brief Destroys a miracle.
 * 
 * @param [in]mrc A miracle.
 * @retval None.
 */
void mrc_destroy_miracle(Miracle *mrc);


/**
 * @brief Prints a miracle.
 * 
 * @param [in]mrc A miracle.
 * @retval None.
 */
void mrc_print_miracle(Miracle *mrc);


/**
 * @brief Assigns literals in a miracle.
 * 
 * @param [in]lits An array of assigned literals.
 * @param [in]lits_len Length of lits, which is the number of assigned
 * literals.
 * @param [in/out]sat_mrc A sat_miracle.
 * @retval None.
 */
void mrc_assign_lits(Lit *lits, int lits_len, sat_miracle *sat_mrc);


/**
 * @brief Increases the decision level in a miracle.
 * 
 * @param [in/out]sat_mrc A sat_miracle.
 * @retval None.
 */
void mrc_increase_decision_level(sat_miracle *sat_mrc);


/**
 * @brief Backjumps to a decision level in a miracle.
 * 
 * @param [in]bj_dec_lvl A backjump decision level. A bj_dec_lvl < 1 resets the
 * miracle.
 * @param [in/out]sat_mrc A sat_miracle.
 * @retval None.
 */
void mrc_backjump(int bj_dec_lvl, sat_miracle *sat_mrc);


/**
 * @brief Computes the RAND heuristic.
 * 
 * @param [in]sat_mrc A sat_miracle.
 * @retval The branching literal.
 */
Lit mrc_RAND_heuristic(sat_miracle *sat_mrc);


/**
 * @brief Computes the JW-OS heuristic.
 * 
 * @param [in]sat_mrc A sat_miracle.
 * @retval The branching literal.
 */
Lit mrc_JW_OS_heuristic(sat_miracle *sat_mrc);


/**
 * @brief Computes the JW-TS heuristic.
 * 
 * @param [in]sat_mrc A sat_miracle.
 * @retval The branching literal.
 */
Lit mrc_JW_TS_heuristic(sat_miracle *sat_mrc);


/**
 * @brief Computes the BOHM heuristic.
 *
 * @param [in]sat_mrc A sat_miracle.
 * @param [in]alpha A constant of the BOHM weight function.
 * @param [in]beta A constant of the BOHM weight function.
 * @retval The branching literal.
 */
Lit mrc_BOHM_heuristic(sat_miracle *sat_mrc, const int alpha, const int beta);


/**
 * @brief Computes the POSIT heuristic.
 * 
 * @param [in]sat_mrc A sat_miracle.
 * @param [in]n A constant of the POSIT weight function.
 * @retval The branching literal.
 */
Lit mrc_POSIT_heuristic(sat_miracle *sat_mrc, const int n);


/**
 * @brief Computes the DLIS heuristic.
 * 
 * @param [in]sat_mrc A sat_miracle.
 * @retval The branching literal.
 */
Lit mrc_DLIS_heuristic(sat_miracle *sat_mrc);


/**
 * @brief Computes the DLCS heuristic.
 * 
 * @param [in]sat_mrc A sat_miracle.
 * @retval The branching literal.
 */
Lit mrc_DLCS_heuristic(sat_miracle *sat_mrc);


/**
 * @brief Computes the RDLIS heuristic.
 * 
 * @param [in]sat_mrc A sat_miracle.
 * @retval The branching literal.
 */
Lit mrc_RDLIS_heuristic(sat_miracle *sat_mrc);


/**
 * @brief Computes the RDLCS heuristic.
 * @param [in]sat_mrc A sat_miracle.
 * @retval The branching literal.
 */
Lit mrc_RDLCS_heuristic(sat_miracle *sat_mrc);


#endif
