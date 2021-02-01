/*
 * miracle.cu
 *
 * file containing the MiraCle library
 *
 */


#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <limits.h>


#include "miracle.h"
#include "formula.h"
#include "utilities.h"


/*
 * global variables
 */

static bool init_PRNG_seed = false;


/*****************************************************************************/


/*
 * API function prototypes
 */

int new_miracle(char *filename);
void delete_miracle();
int DLIS_heuristic();
int DLCS_heuristic();
int RDLIS_heuristic();
int RDLCS_heuristic();
int JW_OS_heuristic();
int JW_TS_heuristic();
int MOM_heuristic(unsigned int k);
int BOHM_heuristic(unsigned int alpha, unsigned int beta);
int RAND_heuristic();


/*****************************************************************************/


/*
 * auxiliary function prototypes
 */

static void init_PRNG();
static int DLxS_heuristic(bool dlcs);
static int RDLxS_heuristic(bool rdlcs);
static int JW_heuristic(bool two_sided);
static double J_l(unsigned int lit);


/*****************************************************************************/


/*
 * API function definitions
 */

/*
 * new_miracle: creates and initializes the oracle MiraCle
 */
int new_miracle(char *filename) {
    int err = new_formula(filename);
    
    if(err) {
        return err;
    }
    
    // DEBUG
    int new_lit_ass_fst[] = {1};
    new_lit_assignments(new_lit_ass_fst, 1);
    int new_lit_ass_snd[] = {-3};
    new_lit_assignments(new_lit_ass_snd, 1);
    int new_lit_ass_trd[] = {2};
    new_lit_assignments(new_lit_ass_trd, 1);
    // int new_lit_ass_frt[] = {33, -18, -45, 94, 58, 56, -65, 30, -48, -77};
    // new_lit_assignments(new_lit_ass_frt, 10);
    // backjumping(1);
    // int new_lit_ass_fft[] = {-72, 20, 68, -54, 59, -57, 89, 98, -96, 83};
    // new_lit_assignments(new_lit_ass_fft, 10);
    // backjumping(0);
    // END DEBUG

    print_formula();

    return 0;
}

/*
 * delete_miracle: destroys the oracle MiraCle previously allocated by new_miracle
 */
void delete_miracle() {
    delete_formula();
}


/*
 * DLIS_heuristic: computes the Dynamic Largest Individual Sum of literals heuristic
 */
int DLIS_heuristic() {
    return DLxS_heuristic(false);
}


/*
 * DLCS_heuristic: computes the Dynamic Largest Combined Sum of literals heuristic
 */
int DLCS_heuristic() {
    return DLxS_heuristic(true);
}


/*
 * RDLIS_heuristic: computes the Random Dynamic Largest Individual Sum of literals heuristic
 */
int RDLIS_heuristic() {
    return RDLxS_heuristic(false);
}


/*
 * RDLCS_heuristic: computes the Random Dynamic Largest Combined Sum of literals heuristic
 */
int RDLCS_heuristic() {
    return RDLxS_heuristic(true);
}


/*
 * JW_OS_heuristic: computes the one-sided Jeroslow-Wang heuristic
 */
int JW_OS_heuristic() {
    return JW_heuristic(false);
}


/*
 * JW_TS_heuristic: computes the two-sided Jeroslow-Wang heuristic
 */
int JW_TS_heuristic() {
    return JW_heuristic(true);
}


/*
 * MOM_heuristic: computes the Maximum Occurrences in clauses of Minimum size
 *                heuristic. Parameter k is a tuning constant present in the
 *                MOM function
 */
int MOM_heuristic(unsigned int k) {
    // finds the minimum size of the clauses
    unsigned int minimum_size = UINT_MAX;
    unsigned int size;

    for(unsigned int c = 1; c <= phi->s_num_clauses; c++) {
        size = phi->d_clause_len[c];

        if(!phi->clause_sat[c] && size < minimum_size) {
            minimum_size = size;
        }
    }

    /*
     * array containing the number of occurrences of literals in clauses of
     * minimum size
     */
    unsigned int *occ_lits;

    HANDLE_ERROR( cudaHostAlloc( (void**) &occ_lits,
                                 sizeof( *occ_lits ) * phi->s_num_vars * 2 + 1,
                                 cudaHostAllocDefault ) );

    // initializes the array elements to 0
    for(unsigned int l = 0; l < phi->s_num_vars * 2 + 1; l++) {
        occ_lits[l] = 0;
    }

    // computes the number of occurrences of literals in clauses of minimum size
    unsigned int lit_start_idx, lit_off, lit_idx;

    for(unsigned int c = 1; c <= phi->s_num_clauses; c++) {
        if(!phi->clause_sat[c] && phi->d_clause_len[c] == minimum_size) {
            lit_start_idx = phi->csr_row_ptr[c];

            for(lit_off = 0; lit_off < phi->s_clause_len[c]; lit_off++) {
                lit_idx = phi->csr_col_ind[lit_start_idx + lit_off];
                
                if(!phi->var_ass[IDX_TO_VAR(lit_idx, phi->s_num_vars)]) {
                    occ_lits[lit_idx]++;
                }
            }
        }
    }

    /*
     * finds the variable x that maximizes the function:
     * f(x) * f(not_x) * 2^k + f(x) + f(not_x)
     */
    unsigned int largest_value = 0, var = 0;
    unsigned int f_pos_v, f_neg_v, value, f_pos_var, f_neg_var;
    
    for(unsigned int v = 1; v <= phi->s_num_vars; v++) {
        if(!phi->var_ass[v]) {
            f_pos_v = occ_lits[v];
            f_neg_v = occ_lits[v + phi->s_num_vars];
            value = f_pos_v * f_neg_v * (1<<k) + f_pos_v + f_neg_v;

            if(value > largest_value) {
                var = v;
                largest_value = value;
                f_pos_var = f_pos_v;
                f_neg_var = f_neg_v;
            }
        }
    }

    return f_pos_var >= f_neg_var ? -var : var;
}


/*
 * BOHM_heuristic: computes the BOHM heuristic
 */
int BOHM_heuristic(unsigned int alpha, unsigned int beta) {
    // array of d_clause_len indices
    unsigned int *asc_idx;

    HANDLE_ERROR( cudaHostAlloc( (void**) &asc_idx,
                                 sizeof( *asc_idx ) * phi->d_num_clauses,
                                 cudaHostAllocDefault ) );

    // initializes asc_idx with unsatisfied clauses
    unsigned int c, i = 0;

    for(c = 1; c <= phi->s_num_clauses; c++) {
        if(!phi->clause_sat[c]) {
            asc_idx[i] = c;
            i++;
        }
    }

    // sorts the d_clause_len array in ascending order using indices in asc_idx
    bool switched = false;
    unsigned int temp;

    do {
        switched = false;

        for(i = 1; i < phi->d_num_clauses; i++) {
            if(phi->d_clause_len[asc_idx[i-1]] > phi->d_clause_len[asc_idx[i]]) {
                temp = asc_idx[i];
                asc_idx[i] = asc_idx[i-1];
                asc_idx[i-1] = temp;
                switched = true;
            }
        }
    } while(switched);

    // array of potential branching variables
    bool *potential_var;

    HANDLE_ERROR( cudaHostAlloc( (void**) &potential_var,
                                 sizeof( *potential_var ) * phi->s_num_vars + 1,
                                 cudaHostAllocDefault ) );
    
    // initializes potential_var considering current assignments
    unsigned int v;

    for(v = 1; v <= phi->s_num_vars; v++) {
        if(!phi->var_ass[v]) {
            potential_var[v] = true;
        } else {
            potential_var[v] = false;
        }
    }

    // array of the number of occurrences of literals in clauses of length i
    unsigned int *h_i;

    HANDLE_ERROR( cudaHostAlloc( (void**) &h_i,
                                 sizeof( *h_i ) * phi->s_num_vars * 2 + 1,
                                 cudaHostAllocDefault ) );
    
    // initializes h_i to 0
    unsigned int l;

    for(l = 1; l <= phi->s_num_vars * 2; l++) {
        h_i[l] = 0;
    }

    // array of the number of occurrences of literals in all clauses
    unsigned int *cum_h_i;

    HANDLE_ERROR( cudaHostAlloc( (void**) &cum_h_i,
                                 sizeof( *cum_h_i ) * phi->s_num_vars * 2 + 1,
                                 cudaHostAllocDefault ) );

    // initializes cum_h_i to 0
    for(l = 1; l <= phi->s_num_vars * 2; l++) {
        cum_h_i[l] = 0;
    }

    // array of H function values for variables
    unsigned int *H_i;

    HANDLE_ERROR( cudaHostAlloc( (void**) &H_i,
                                 sizeof( *H_i ) * phi->s_num_vars + 1,
                                 cudaHostAllocDefault ) );
    
    // initializes H_i to 0
    for(v = 1; v <= phi->s_num_vars; v++) {
        H_i[v] = 0;
    }

    /*
     * finds the literal l having the maximal vector:
     * (H_1(l), H_2(l), ..., H_n(l)) in lexicographic order
     */
    unsigned int clause_idx, lit_start_idx, lit_off, lit_idx, var;
    unsigned int h_i_pos_v, h_i_neg_v, max_h_i, min_h_i;
    unsigned int value;
    unsigned int largest_value = 0, count_pot_var = 0, pot_var = 0;
    bool found_var = false;

    for(i = 0; i < phi->d_num_clauses; i++) {
        // increments h_i and cum_h_i
        clause_idx = asc_idx[i];
        lit_start_idx = phi->csr_row_ptr[clause_idx];

        for(lit_off = 0; lit_off < phi->s_clause_len[clause_idx]; lit_off++) {
            lit_idx = phi->csr_col_ind[lit_start_idx + lit_off];
            var = IDX_TO_VAR(lit_idx, phi->s_num_vars);
            
            if(potential_var[var]) {
                h_i[lit_idx]++;
                cum_h_i[lit_idx]++;
            }
        }

        if(!found_var &&
           (i == phi->d_num_clauses - 1 ||
            phi->d_clause_len[clause_idx] < phi->d_clause_len[asc_idx[i+1]]
           )
          ) {
            // computes H_i
            for(v = 1; v <= phi->s_num_vars; v++) {
                if(potential_var[v]) {
                    h_i_pos_v = h_i[v];
                    h_i_neg_v = h_i[v + phi->s_num_vars];
                    max_h_i = h_i_pos_v >= h_i_neg_v ? h_i_pos_v : h_i_neg_v;
                    min_h_i = h_i_pos_v < h_i_neg_v ? h_i_pos_v : h_i_neg_v;
                    H_i[v] = alpha * max_h_i + beta * min_h_i;
                }
            }

            // finds the number of variables with the largest H_i value
            for(v = 1; v <= phi->s_num_vars; v++) {
                if(potential_var[v]) {
                    value = H_i[v];

                    if(value > largest_value) {
                        largest_value = value;
                        count_pot_var = 1;
                        pot_var = v;
                    } else if (value == largest_value) {
                        count_pot_var++;
                    }
                }
            }

            if(count_pot_var >= 2 && i < phi->d_num_clauses - 1) {
                // updates potential_var
                for(v = 1; v <= phi->s_num_vars; v++) {
                    if(potential_var[v] && H_i[v] < largest_value) {
                        potential_var[v] = false;
                    }
                }

                // resets H_i to 0
                for(v = 1; v <= phi->s_num_vars; v++) {
                    H_i[v] = 0;
                }

                // resets h_i to 0
                for(l = 1; l <= phi->s_num_vars * 2; l++) {
                    h_i[l] = 0;
                }

                /*
                 * resets count_pot_var and largest_value. It doesn't reset
                 * pot_var because it may not be updated anymore
                 */
                count_pot_var = largest_value = 0;
            } else {
                // updates found_var and potential_var
                found_var = true;

                for(v = 1; v <= phi->s_num_vars; v++) {
                    if(potential_var[v] && v != pot_var) {
                        potential_var[v] = false;
                    }
                }
            }
        }
    }

    return cum_h_i[pot_var] >= cum_h_i[pot_var + phi->s_num_vars] ? pot_var : -pot_var;
}


/*
 * RAND_heuristic: computes the RAND heuristic
 */
int RAND_heuristic() {
    init_PRNG();

    unsigned int v, var = 0;
    int random = rand() % phi->d_num_vars;
    int i = 0;

    for(v = 1; v <= phi->s_num_vars; v++) {
        if(!phi->var_ass[v]) {
            if(i < random) {
                i++;
            } else {
                var = v;
                break;
            }
        }
    }

    if(rand() % 2) {
        return -var;
    } else {
        return var;
    }
}


/*****************************************************************************/


/*
 * auxiliary function definitions
 */

/*
 * init_PRNG: initializes PRNG with a seed
 */
static void init_PRNG() {
    if(!init_PRNG_seed) {
        srand(time(NULL));
        init_PRNG_seed = true;
    }
}


/*
 * DLxS_heuristic: if dlcs is true, it computes the DLCS heuristic, otherwise
 *                 it computes the DLIS heuristic
 */
static int DLxS_heuristic(bool dlcs) {
    unsigned int var = 0;
    unsigned int largest_sum = 0;
    unsigned int var_cp, var_cn, sum, v_cp, v_cn;

    for(unsigned int v = 1; v <= phi->s_num_vars; v++) {
        if(!phi->var_ass[v]) {
            v_cp = phi->d_occ_lits[v];
            v_cn = phi->d_occ_lits[v + phi->s_num_vars];

            if(dlcs) {
                sum = v_cp + v_cn;
            } else {
                sum = v_cp >= v_cn ? v_cp : v_cn;
            }

            if(sum > largest_sum) {
                var = v;
                largest_sum = sum;
                var_cp = v_cp;
                var_cn = v_cn;
            }
        }
    }

    return var_cp >= var_cn ? var : -var;
}


/*
 * RDLxS_heuristic: if rdlcs is true, it computes the RDLCS heuristic, otherwise
 *                  it computes the RDLIS heuristic
 */
static int RDLxS_heuristic(bool rdlcs) {
    init_PRNG();

    if(rdlcs && (rand() % 2)) {
        return -DLCS_heuristic();
    } else if(rdlcs) {
        return DLCS_heuristic();
    } else if(rand() % 2) {
        return -DLIS_heuristic();
    } else {
        return DLIS_heuristic();
    }
}


/*
 * JW_heuristic: if two_sided is true, it computes the JW-TS heuristic, otherwise
 *               it computes the JW-OS heuristic
 */
static int JW_heuristic(bool two_sided) {
    unsigned int var = 0;
    double largest_value = 0.0;
    double J_pos_var, J_neg_var, value, J_pos_v, J_neg_v;

    for(unsigned int v = 1; v <= phi->s_num_vars; v++) {
        if(!phi->var_ass[v]) {
            J_pos_v = J_l(v);
            J_neg_v = J_l(v + phi->s_num_vars);

            if(two_sided) {
                value = J_pos_v + J_neg_v;
            } else {
                value = J_pos_v >= J_neg_v ? J_pos_v : J_neg_v;
            }

            if(value > largest_value) {
                var = v;
                largest_value = value;
                J_pos_var = J_pos_v;
                J_neg_var = J_neg_v;
            }
        }
    }

    return J_pos_var >= J_neg_var ? var : -var;
}


/*
 * J_l: computes the J function of the Jeroslow-Wang heuristic
 */
static double J_l(unsigned int lit) {
    unsigned int clause_start_idx = phi->csc_col_ptr[lit];
    unsigned int clause_idx;
    double J_lit = 0.0;

    for(unsigned int clause_off = 0; clause_off < phi->s_occ_lits[lit]; clause_off++) {
        clause_idx = phi->csc_row_ind[clause_start_idx + clause_off];

        if(!(phi->clause_sat[clause_idx])) {
            J_lit += pow(2.0, -double(phi->d_clause_len[clause_idx]));
        }
    }

    return J_lit;
}


/*****************************************************************************/
