/*
 * formula.h
 *
 * file containing the formula library API
 *
 */


#ifndef FORMULA
#define FORMULA


int new_formula(char *filename);
void delete_formula();
void print_formula();
void new_lit_assignments(int *new_lit_ass, unsigned int new_lit_ass_len);
void backjumping(unsigned int bj_dec_lvl);


#endif
