/*
 * miracle.h
 *
 * file containing the MiraCle library API
 *
 */


#ifndef MIRACLE
#define MIRACLE


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


#endif
