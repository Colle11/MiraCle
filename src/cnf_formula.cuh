/**
 * cnf_formula.cuh: definition of the CNF_Formula data type and declaration of
 * its API.
 * 
 * Copyright (c) Michele Collevati
 */


#ifndef CNF_FORMULA_CUH
#define CNF_FORMULA_CUH


#include "cnf_formula_types.cuh"


/**
 * @brief CNF_Formula data type.
 */
typedef struct cnf_formula {
    Lidx *clauses;              // Array of clauses.
    int clauses_len;            // Length of clauses, which is num_lits.
    
    int *clause_indices;        // Array of clause indices.
    int clause_indices_len;     /**
                                 * Length of clause_indices, which is
                                 * num_clauses+1.
                                 */

    int num_vars;               // Number of variables.
    int num_clauses;            // Number of clauses.
    int num_lits;               // Number of literals.
} CNF_Formula;


/**
 * API
 */


/**
 * @brief Parses a formula from a file in DIMACS CNF format.
 * 
 * @param [in]filename Filename of a formula in DIMACS CNF format.
 * @retval An initialized formula.
 */
CNF_Formula *cnf_parse_DIMACS(char *filename);


/**
 * @brief Destroys a formula.
 * 
 * @param [in]phi A formula.
 * @retval None.
 */
void cnf_destroy_formula(CNF_Formula *phi);


/**
 * @brief Prints a formula.
 * 
 * @param [in]phi A formula.
 * @retval None.
 */
void cnf_print_formula(CNF_Formula *phi);


#endif
