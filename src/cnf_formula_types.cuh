/**
 * cnf_formula_types.cuh: definition of CNF_Formula types and their API.
 *
 * Copyright (c) Michele Collevati
 */


#ifndef CNF_FORMULA_TYPES_CUH
#define CNF_FORMULA_TYPES_CUH


#include <stdlib.h>


/**
 * Types
 */


/**
 * Variables are integers from 0 to N, so that they can be used as array
 * indices.
 */
typedef int Var;
#define UNDEF_VAR (-1)


/**
 * Literals are integers from -(N + 1) to N + 1, 0 excluded.
 */
typedef int Lit;
#define UNDEF_LIT (0)


/**
 * Literal indices are integers from 0 to 2N + 1, so that they can be used as
 * array indices.
 */
typedef int Lidx;
#define UNDEF_LIDX (-1)


/**
 * API
 */


/**
 * @brief Gets the literal index from a variable and a polarity.
 * 
 * @param [in]var A variable.
 * @param [in]polarity A polarity.
 * @retval The literal index.
 */
inline __host__ __device__ Lidx varpol_to_lidx(Var var, bool polarity) {
    return (Lidx)(var + var + (Var)(!polarity));
}


inline __host__ __device__ Lit varpol_to_lit(Var var, bool polarity) {
    return (Lit)(polarity ? var + 1 : -(var + 1));
}


/**
 * @brief Gets the literal index from a literal.
 * 
 * @param [in]lit A literal.
 * @retval The literal index.
 */
inline __host__ __device__ Lidx lit_to_lidx(Lit lit) {
    return (Lidx)(abs(lit + lit) - 2 + (Lit)(lit < 0));
}


/**
 * @brief Gets the variable from a literal.
 * 
 * @param [in]lit A literal.
 * @retval The variable.
 */
inline __host__ __device__ Var lit_to_var(Lit lit) {
    return (Var)(abs(lit) - 1);
}


/**
 * @brief Gets the polarity from a literal.
 * 
 * @param [in]lit A literal.
 * @retval The polarity (true/positive or false/negative).
 */
inline __host__ __device__ bool lit_to_pol(Lit lit) {
    return lit > 0;
}


/**
 * @brief Negates a literal.
 * 
 * @param [in]lit A literal.
 * @retval The negated literal.
 */
inline __host__ __device__ Lit neg_lit(Lit lit) {
    return -lit;
}


/**
 * @brief Gets the variable from a literal index.
 * 
 * @param [in]lidx A literal index.
 * @retval The variable.
 */
inline __host__ __device__ Var lidx_to_var(Lidx lidx) {
    return (Var)(lidx >> 1);
}


/**
 * @brief Gets the polarity from a literal index.
 * 
 * @param [in]lidx A literal index.
 * @retval The polarity (true/positive or false/negative).
 */
inline __host__ __device__ bool lidx_to_pol(Lidx lidx) {
    return !(lidx & 1);
}


/**
 * @brief Gets the literal from a literal index.
 * 
 * @param [in]lidx A literal index.
 * @retval The literal.
 */
inline __host__ __device__ Lit lidx_to_lit(Lidx lidx) {
    return (Lit)(lidx & 1 ? -((lidx >> 1) + 1) : (lidx >> 1) + 1);
}


/**
 * @brief Negates a literal index.
 * 
 * @param [in]lidx A literal index.
 * @retval The negated literal index.
 */
inline __host__ __device__ Lidx neg_lidx(Lidx lidx) {
    return lidx & 1 ? lidx - 1 : lidx + 1;
}


#endif
