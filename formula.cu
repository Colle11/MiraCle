/*
 * formula.cu
 *
 * file containing the implementation of the formula data type API
 * (formula data type library)
 *
 */

/*
 * function prototypes
 */

formula *alloc_formula();

/*
 * function definitions
 */

/*
 * alloc_formula: allocates an empty formula
 */
formula *alloc_formula() {
    formula *f;

    HANDLE_ERROR( cudaHostAlloc( (void**) &f,
                                 sizeof( *f ),
                                 cudaHostAllocDefault ) );
    
    return f;
}
