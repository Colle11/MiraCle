/**
 * cnf_formula.cu: definition of the CNF_Formula API.
 * 
 * Copyright (c) Michele Collevati
 */


#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>


#include "cnf_formula.cuh"
#include "utils.cuh"


// #define CHECK_VARS_IN_CLAUSE


#define AVG_NUM_LITS_IN_CLAUSE (3)    // Average number of literals in a clause.


#ifdef CHECK_VARS_IN_CLAUSE
bool repeated_var_in_clause = false;
#endif


/**
 * Auxiliary function prototypes
 */


/**
 * @brief Skips a line in a file.
 * 
 * @param [in]fp A file pointer.
 * @retval None.
 */
static void skip_line(FILE *fp);


/**
 * @brief Parses the problem line in a file in DIMACS CNF format.
 * 
 * @param [in]fp A DIMACS CNF file pointer.
 * @param [out]num_vars The number of variables declared in the problem line.
 * @param [out]num_clauses The number of clauses declared in the problem line.
 * @retval None.
 */
static void parse_problem_line(FILE *fp,
                               int *num_vars,
                               int *num_clauses);


/**
 * @brief Creates an empty formula.
 * 
 * @param [in]num_clauses A number of clauses.
 * @param [in]est_num_lits An estimate of the number of literals.
 * @retval An empty formula.
 */
static CNF_Formula *create_formula(int num_clauses,
                                   int est_num_lits);


/**
 * @brief Parses a clause line in a file in DIMACS CNF format.
 * 
 * @param [in]fp A DIMACS CNF file pointer.
 * @param [in/out]est_num_lits An estimate of the number of literals.
 * @param [in/out]phi A formula being initialized.
 * @retval None.
 */
static void parse_clause_line(FILE *fp,
                              int *est_num_lits,
                              CNF_Formula *phi);


/**
 * API definition
 */


CNF_Formula *cnf_parse_DIMACS(char *filename) {
    FILE *fp = fopen(filename, "r");

    if (fp == NULL) {
        fprintf(stderr, "Can't open %s.\n", filename);
        exit(EXIT_FAILURE);
    }

    int c;      // Currently read character.
    bool p_ln_found = false;    /**
                                 * Flag to check if the problem line has been
                                 * found.
                                 */
    int p_num_vars;             /**
                                 * Number of variables declared in the problem
                                 * line.
                                 */
    int p_num_clauses;          /**
                                 * Number of clauses declared in the problem
                                 * line.
                                 */
    int est_num_lits;           // Estimate of the number of literals.
    CNF_Formula *phi;

    while (!feof(fp)) {
        c = fgetc(fp);

        if (c == 'c') {     // Skip a comment line.
            skip_line(fp);
        } else if (c == 'p') {      // Parse the problem line.
            p_ln_found = true;

            parse_problem_line(fp, &p_num_vars, &p_num_clauses);

            est_num_lits = p_num_clauses * AVG_NUM_LITS_IN_CLAUSE;

            phi = create_formula(p_num_clauses, est_num_lits);
        } else if (('1' <= c && c <= '9') || c == '-') {
            if (!p_ln_found) {
                fprintf(stderr, "The problem line is missing in the DIMACS "
                        "CNF file \"%s\".\n", filename);
                exit(EXIT_FAILURE);
            }

            ungetc(c, fp);

            // Parse a clause line.
            parse_clause_line(fp, &est_num_lits, phi);
        }
    }

    int num_lits = phi->num_lits;
    int num_clauses = phi->num_clauses;

    phi->clause_indices[num_clauses] = num_lits;

    // Adapt the memory of phi->clauses.
    phi->clauses = (Lidx *)realloc(phi->clauses,
                                   sizeof *(phi->clauses) * num_lits);
    phi->clauses_len = num_lits;

    fclose(fp);

    // Check the correctness of the problem line.
    if (p_num_vars != phi->num_vars || p_num_clauses != num_clauses) {
        fprintf(stderr, "The problem line of the DIMACS CNF file \"%s\" is "
                "incorrect.\n", filename);
        exit(EXIT_FAILURE);
    }

#ifdef CHECK_VARS_IN_CLAUSE
    if (repeated_var_in_clause) {
        fprintf(stderr, "Found repeated variables in the clauses of the "
                "DIMACS CNF file \"%s\".\n", filename);
        exit(EXIT_FAILURE);
    } else {
        // exit(EXIT_SUCCESS);
    }
#endif

    return phi;
}


void cnf_destroy_formula(CNF_Formula *phi) {
    free(phi->clauses);
    free(phi->clause_indices);
    free(phi);
}


void cnf_print_formula(CNF_Formula *phi) {
    printf("*** CNF formula ***\n\n");

    printf("Number of variables: %d\n", phi->num_vars);
    printf("Number of clauses: %d\n", phi->num_clauses);
    printf("Number of literals: %d\n", phi->num_lits);

    printf("Clause indices: ");
    int clause_indices_len = phi->clause_indices_len;
    int *clause_indices = phi->clause_indices;
    for (int i = 0; i < clause_indices_len; i++) {
        printf("[%d]%d ", i, clause_indices[i]);
    }
    printf("\n");

    printf("Clauses: ");
    int clauses_len = phi->clauses_len;
    Lidx *clauses = phi->clauses;
    for (int l = 0; l < clauses_len; l++) {
        printf("[%d]%d ", l, clauses[l]);
    }
    printf("\n");

    printf("\n*** End CNF formula ***\n\n");
}


/**
 * Auxiliary function definitions
 */


static void skip_line(FILE *fp) {
    int c = fgetc(fp);

    while (c != '\n') {
        c = fgetc(fp);
    }
}


static void parse_problem_line(FILE *fp,
                               int *num_vars,
                               int *num_clauses) {
    fscanf(fp, "%*s %d %d", num_vars, num_clauses);

    int c = fgetc(fp);

    while (c != '\n') {
        c = fgetc(fp);
    }
}


static CNF_Formula *create_formula(int num_clauses,
                                   int est_num_lits) {
    CNF_Formula *phi = (CNF_Formula *)malloc(sizeof *phi);

    phi->clause_indices_len = num_clauses + 1;
    phi->clause_indices = (int *)malloc(sizeof *(phi->clause_indices) *
                                        phi->clause_indices_len);

    phi->clauses_len = est_num_lits;
    phi->clauses = (Lidx *)malloc(sizeof *(phi->clauses) * phi->clauses_len);

    phi->num_vars = 0;
    phi->num_clauses = 0;
    phi->num_lits = 0;

    return phi;
}


static void parse_clause_line(FILE *fp,
                              int *est_num_lits,
                              CNF_Formula *phi) {
    phi->clause_indices[phi->num_clauses] = phi->num_lits;
    phi->num_clauses++;

    bool polarity = true;
    Var var = 0;
    int c = fgetc(fp);

#ifdef CHECK_VARS_IN_CLAUSE
    int num_lits_in_clause = 0;
#endif

    while (c != '0') {
        if (c == '-') {
            polarity = false;
        } else {    // '1' <= c && c <= '9'
            var = c - '0';
        }

        c = fgetc(fp);

        while ('0' <= c && c <= '9') {      // Parse a literal.
            var *= 10;
            var += c - '0';
            c = fgetc(fp);
        }

#ifdef CHECK_VARS_IN_CLAUSE
        for (int l = 1; l <= num_lits_in_clause; l++) {
            if (var - 1 == lidx_to_var(phi->clauses[phi->num_lits - l])) {
                fprintf(stderr,
                        "Clause: %d\tVar: %d\n",
                        phi->num_clauses,
                        var);
                repeated_var_in_clause = true;
            }
        }

        num_lits_in_clause++;
#endif

        phi->num_vars = max(phi->num_vars, var);

        // Increase the memory of phi->clauses.
        if (phi->num_lits == *est_num_lits) {
            *est_num_lits *= 2;
            
            phi->clauses = (Lidx *)realloc(phi->clauses,
                                           sizeof *(phi->clauses) *
                                           (*est_num_lits));
            phi->clauses_len = *est_num_lits;
        }

        phi->clauses[phi->num_lits] = varpol_to_lidx(var - 1, polarity);
        phi->num_lits++;

        var = 0;
        polarity = true;

        while (!('1' <= c && c <= '9') && c != '-' && c != '0') {
            c = fgetc(fp);
        }
    }
}
