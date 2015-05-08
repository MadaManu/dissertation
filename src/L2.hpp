#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <stdlib.h>
#include <string>
#include <x86intrin.h>
#include <stdio.h>
#include <bitset>
#include <time.h>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <config.h>
    /*** LEVEL 2 ***/
// TODO: implement same using f48 maybe to compare some timmings
// TODO: function to take SIZE
// function is void as should overwrite the input
// works for square matrix - ?? does it require to work for not square matrix?
// ^^ for non-square matrix requires computation of final SIZE of vector
// matrix * vector always returns vector (SIZE is dependant on the matrix SIZE
void matrix_vector_mul_double(double** mat, double* &vec)
{
  double* result = new double[SIZE];
  for(unsigned i=0;i<SIZE;i++) { // row
    double running_sum = 0;
    for(unsigned j=0;j<SIZE;j++) { // col
	running_sum += mat[i][j]*vec[j];
    }
    result[i] = running_sum;
  }
  vec = result;
}

void matrix_vector_mul_f48(f48** mat, f48* &vec)
{
  f48* result = new f48[SIZE];
  for(unsigned i=0;i<SIZE;i++) { // row
    double running_sum = 0;
    for(unsigned j=0;j<SIZE;j++) { // col
	running_sum += double(mat[i][j])*double(vec[j]);
    }
    result[i] = (f48)running_sum;
  }
  vec = result;
}

// TODO: function to take SIZE
// TODO: requires implementation
// matrix is square?
// same questions as above
void matrix_vector_mul_SSE_double(double** mat, double* &vec)
{
  double* result = new double[SIZE]; // should be SIZE of result!
  for(unsigned i=0;i<SIZE;i++) { // row
    __m128d running_sum = _mm_set1_pd(0.0); // running sum initially 0
    for(unsigned j=0;j<SIZE;j+=2) { // col - requires skipping on 2 at a time
      // multiply each
      // add to running sum
      __m128d mat_vect = _mm_load_pd(&mat[i][j]); // hoping that addresses are as expected - seems like this is the way it's stored
						  // ^^ needs explanation and backup for REPORT - ROW major storing order in C/C++ such as python, pascal and others
      __m128d vec_elem = _mm_load_pd(&vec[j]);
      __m128d mult = _mm_mul_pd(mat_vect,vec_elem);
      running_sum = _mm_add_pd(mult,running_sum);


    }
    // shuffle & add (to make hadd)
    // store back to vec[i]
    __m128i mask = _mm_set_epi8(7 ,6 ,5, 4, 3, 2, 1, 0,
		      15, 14, 13, 12, 11, 10, 9, 8);
    __m128i sum_shuffled = _mm_shuffle_epi8((__m128i)running_sum, mask);
    running_sum = _mm_add_pd(running_sum,(__m128d)sum_shuffled);
    // convert running_sum back to f48s and store in memory

    _mm_store_sd(&result[i], running_sum);
  }
  vec = result;
}

// f48 V1
void matrix_vector_mul_SSE_f48(f48** mat, f48* &vec)
{
  f48* result = new f48[SIZE]; // should be SIZE of result!
  __m128i mask = _mm_set_epi8(11, 10, 9, 8, 7, 6, 255, 255,
  			      5, 4, 3, 2, 1, 0, 255, 255);
  __m128i shuffling_mask = _mm_set_epi8(7 ,6 ,5, 4, 3, 2, 1, 0,
			      15, 14, 13, 12, 11, 10, 9, 8);
  for(unsigned i=0;i<SIZE;i++) { // row
    __m128d running_sum = _mm_set1_pd(0.0); // running sum initially 0
    for(unsigned j=0;j<SIZE;j+=2) { // col - requires skipping on 2 at a time

      // multiply each
      // add to running sum
      __m128i mat_vect = _mm_loadu_si128((__m128i*) &mat[i][j]); // hoping that addresses are as expected - seems like this is the way it's stored
						  // ^^ needs explanation and backup for REPORT - ROW major storing order in C/C++ such as python, pascal and others
      mat_vect = _mm_shuffle_epi8(mat_vect, mask);
      __m128i vec_elem = _mm_loadu_si128((__m128i*) &vec[j]);
      vec_elem = _mm_shuffle_epi8(vec_elem, mask);

      __m128d mult = _mm_mul_pd((__m128d)mat_vect,(__m128d)vec_elem);
      running_sum = _mm_add_pd(mult,running_sum);


    }
    // shuffle & add (to make hadd)
    // store back to vec[i]
    __m128i sum_shuffled = _mm_shuffle_epi8((__m128i)running_sum, shuffling_mask);
    running_sum = _mm_add_pd(running_sum,(__m128d)sum_shuffled);
    double temp=0;
    _mm_store_sd(&temp, running_sum);
    result[i]=f48(temp);
  }
  vec = result;
}

// v2 is taking 4 at a time
//
//    mat        vec
// |a|b|c|d|     |W|
// |e|f|g|h|     |X|
// |i|j|k|l|     |Y|
// |m|n|o|p|     |Z|
//
// Function computes: - WRONG - requires UPDATE!
// a*W+b*W+c*W+d*W | e*X+f*X+g*X+h*X and stores them @ &vec[0] (storing first result in vec[0] & second result in vec[1]
// i*Y+j*Y+k*Y+l*Y | m*Z+n*Z+o*Z+p*Z and stores them @ &vec[2] (storing first result in vec[2] & second result in vec[3]
//
//
// ^^ TODO: FIX THIS EXPLANATION AND DO SIMILAR FOR EACH FUNCTION IN THE IMPLEMENTATION!!!!
//
void matrix_vector_mul_SSE_double_v2(double** mat, double* &vec)
{
  double* result = new double[SIZE];
  for(unsigned i=0;i<SIZE;i+=2) { // row // requiring 2 at a time
    __m128d running_sum1 = _mm_set1_pd(0.0); // running sum initially 0
    __m128d running_sum2 = _mm_set1_pd(0.0); // running sum initially 0
    for(unsigned j=0;j<SIZE;j+=2) { // col - requires skipping on 2 at a time
       __m128d mat_vect = _mm_load_pd(&mat[i][j]); // hoping that addresses are as expected - seems like this is the way it's stored
						  // ^^ needs explanation and backup for REPORT TODO
      __m128d vec_elem = _mm_load_pd(&vec[j]);
      __m128d mult = _mm_mul_pd(mat_vect,vec_elem);
      running_sum1 = _mm_add_pd(mult,running_sum1);

       mat_vect = _mm_load_pd(&mat[i+1][j]); // hoping that addresses are as expected - seems like this is the way it's stored
						  // ^^ needs explanation and backup for REPORT TODO
      mult = _mm_mul_pd(mat_vect,vec_elem);
      running_sum2 = _mm_add_pd(mult,running_sum2);
    }
    __m128i mask = _mm_set_epi8(7 ,6 ,5, 4, 3, 2, 1, 0,
		      15, 14, 13, 12, 11, 10, 9, 8);
    __m128i sum_shuffled = _mm_shuffle_epi8((__m128i)running_sum1, mask);
    running_sum1 = _mm_add_pd(running_sum1,(__m128d)sum_shuffled);
    sum_shuffled = _mm_shuffle_epi8((__m128i)running_sum2, mask);
    running_sum2 = _mm_add_pd(running_sum2,(__m128d)sum_shuffled);
    // shuffling so both results can be stored in one store instruction
    mask = _mm_set_epi8(255, 255, 255, 255, 255, 255, 255, 255,
			7 ,6 ,5, 4, 3, 2, 1, 0);
    running_sum1 = (__m128d)_mm_shuffle_epi8((__m128i)running_sum1, mask);
    mask = _mm_set_epi8(7 ,6 ,5, 4, 3, 2, 1, 0,
			255, 255, 255, 255, 255, 255, 255, 255);
    running_sum2 = (__m128d)_mm_shuffle_epi8((__m128i)running_sum2, mask);
    running_sum1 = (__m128d)_mm_or_si128((__m128i)running_sum1,(__m128i)running_sum2);
    _mm_store_pd(&result[i], running_sum1);
  }
  vec = result;
}

void matrix_vector_mul_SSE_f48_v2(f48** mat, f48* &vec)
{
  f48* result = new f48[SIZE];
  __m128i load_mask = _mm_set_epi8(11, 10, 9, 8, 7, 6, 255, 255,
  			      5, 4, 3, 2, 1, 0, 255, 255);
  for(unsigned i=0;i<SIZE;i+=2) { // row // requiring 2 at a time
    __m128d running_sum1 = _mm_set1_pd(0.0); // running sum initially 0
    __m128d running_sum2 = _mm_set1_pd(0.0); // running sum initially 0
    for(unsigned j=0;j<SIZE;j+=2) { // col - requires skipping on 2 at a time
      __m128i mat_vect = _mm_loadu_si128((__m128i*) &mat[i][j]); // hoping that addresses are as expected - seems like this is the way it's stored
						  // ^^ needs explanation and backup for REPORT - ROW major storing order in C/C++ such as python, pascal and others
      mat_vect = _mm_shuffle_epi8(mat_vect, load_mask);
      __m128i vec_elem = _mm_loadu_si128((__m128i*) &vec[j]);
      vec_elem = _mm_shuffle_epi8(vec_elem, load_mask);

      __m128d mult = _mm_mul_pd((__m128d)mat_vect,(__m128d)vec_elem);
      running_sum1 = _mm_add_pd(mult,running_sum1);

      mat_vect = _mm_loadu_si128((__m128i*) &mat[i+1][j]); // hoping that addresses are as expected - seems like this is the way it's stored
						  // ^^ needs explanation and backup for REPORT - ROW major storing order in C/C++ such as python, pascal and others
      mat_vect = _mm_shuffle_epi8(mat_vect, load_mask);

      mult = _mm_mul_pd((__m128d)mat_vect,(__m128d)vec_elem);
      running_sum2 = _mm_add_pd(mult,running_sum2);
    }
    __m128i mask = _mm_set_epi8(7 ,6 ,5, 4, 3, 2, 1, 0,
		      15, 14, 13, 12, 11, 10, 9, 8);
    __m128i sum_shuffled = _mm_shuffle_epi8((__m128i)running_sum1, mask);
    running_sum1 = _mm_add_pd(running_sum1,(__m128d)sum_shuffled);
    sum_shuffled = _mm_shuffle_epi8((__m128i)running_sum2, mask);
    running_sum2 = _mm_add_pd(running_sum2,(__m128d)sum_shuffled);
    double temp = 0;
    _mm_store_sd((double*)&temp, running_sum2);
    result[i] = f48(temp);
    _mm_store_sd((double*)&temp, running_sum2);
    result[i+1] = f48(temp);
  }
  vec = result;
}

void matrix_vector_mul_SSE_f48_loop_unrolled (f48** mat, f48* &vec)
{
    // TESTING change SIZE to min 8 - but multiple of 8
    f48* result = new f48[SIZE];
  __m128i load_mask = _mm_set_epi8(11, 10, 9, 8, 7, 6, 255, 255,
  			      5, 4, 3, 2, 1, 0, 255, 255);
  for(unsigned i=0;i<SIZE;i+=8) { // row // requiring 8 at a time - because loop un-roll
    __m128d running_sum1 = _mm_set1_pd(0.0); // running sum initially 0
    __m128d running_sum2 = _mm_set1_pd(0.0); // running sum initially 0
    __m128d running_sum3 = _mm_set1_pd(0.0); // running sum initially 0
    __m128d running_sum4 = _mm_set1_pd(0.0); // running sum initially 0
    __m128d running_sum5 = _mm_set1_pd(0.0); // running sum initially 0
    __m128d running_sum6 = _mm_set1_pd(0.0); // running sum initially 0
    __m128d running_sum7 = _mm_set1_pd(0.0); // running sum initially 0
    __m128d running_sum8 = _mm_set1_pd(0.0); // running sum initially 0

    for(unsigned j=0;j<SIZE;j+=2) { // col - requires skipping on 2 at a time
      __m128i mat_vect = _mm_loadu_si128((__m128i*) &mat[i][j]); // hoping that addresses are as expected - seems like this is the way it's stored
      mat_vect = _mm_shuffle_epi8(mat_vect, load_mask);
      __m128i vec_elem = _mm_loadu_si128((__m128i*) &vec[j]);
      vec_elem = _mm_shuffle_epi8(vec_elem, load_mask);
      __m128d mult = _mm_mul_pd((__m128d)mat_vect,(__m128d)vec_elem);
      running_sum1 = _mm_add_pd(mult,running_sum1);

      mat_vect = _mm_loadu_si128((__m128i*) &mat[i+1][j]);
      mat_vect = _mm_shuffle_epi8(mat_vect, load_mask);
      mult = _mm_mul_pd((__m128d)mat_vect,(__m128d)vec_elem);
      running_sum2 = _mm_add_pd(mult,running_sum2);

      mat_vect = _mm_loadu_si128((__m128i*) &mat[i+2][j]);
      mat_vect = _mm_shuffle_epi8(mat_vect, load_mask);
      mult = _mm_mul_pd((__m128d)mat_vect,(__m128d)vec_elem);
      running_sum3 = _mm_add_pd(mult,running_sum3);

      mat_vect = _mm_loadu_si128((__m128i*) &mat[i+3][j]);
      mat_vect = _mm_shuffle_epi8(mat_vect, load_mask);
      mult = _mm_mul_pd((__m128d)mat_vect,(__m128d)vec_elem);
      running_sum4 = _mm_add_pd(mult,running_sum4);

      mat_vect = _mm_loadu_si128((__m128i*) &mat[i+4][j]);
      mat_vect = _mm_shuffle_epi8(mat_vect, load_mask);
      mult = _mm_mul_pd((__m128d)mat_vect,(__m128d)vec_elem);
      running_sum5 = _mm_add_pd(mult,running_sum5);

      mat_vect = _mm_loadu_si128((__m128i*) &mat[i+5][j]);
      mat_vect = _mm_shuffle_epi8(mat_vect, load_mask);
      mult = _mm_mul_pd((__m128d)mat_vect,(__m128d)vec_elem);
      running_sum6 = _mm_add_pd(mult,running_sum6);

      mat_vect = _mm_loadu_si128((__m128i*) &mat[i+6][j]);
      mat_vect = _mm_shuffle_epi8(mat_vect, load_mask);
      mult = _mm_mul_pd((__m128d)mat_vect,(__m128d)vec_elem);
      running_sum7 = _mm_add_pd(mult,running_sum7);

      mat_vect = _mm_loadu_si128((__m128i*) &mat[i+7][j]);
      mat_vect = _mm_shuffle_epi8(mat_vect, load_mask);
      mult = _mm_mul_pd((__m128d)mat_vect,(__m128d)vec_elem);
      running_sum8 = _mm_add_pd(mult,running_sum8);
    }
    __m128i mask = _mm_set_epi8(7 ,6 ,5, 4, 3, 2, 1, 0,
		      15, 14, 13, 12, 11, 10, 9, 8);
    __m128i sum_shuffled = _mm_shuffle_epi8((__m128i)running_sum1, mask);
    running_sum1 = _mm_add_pd(running_sum1,(__m128d)sum_shuffled);
    sum_shuffled = _mm_shuffle_epi8((__m128i)running_sum2, mask);
    running_sum2 = _mm_add_pd(running_sum2,(__m128d)sum_shuffled);
    sum_shuffled = _mm_shuffle_epi8((__m128i)running_sum3, mask);
    running_sum3 = _mm_add_pd(running_sum3,(__m128d)sum_shuffled);
    sum_shuffled = _mm_shuffle_epi8((__m128i)running_sum4, mask);
    running_sum4 = _mm_add_pd(running_sum4,(__m128d)sum_shuffled);
    sum_shuffled = _mm_shuffle_epi8((__m128i)running_sum5, mask);
    running_sum5 = _mm_add_pd(running_sum5,(__m128d)sum_shuffled);
    sum_shuffled = _mm_shuffle_epi8((__m128i)running_sum6, mask);
    running_sum6 = _mm_add_pd(running_sum6,(__m128d)sum_shuffled);
    sum_shuffled = _mm_shuffle_epi8((__m128i)running_sum7, mask);
    running_sum7 = _mm_add_pd(running_sum7,(__m128d)sum_shuffled);
    sum_shuffled = _mm_shuffle_epi8((__m128i)running_sum8, mask);
    running_sum8 = _mm_add_pd(running_sum8,(__m128d)sum_shuffled);

    // mesh them into 4
    __m128i mask_first = _mm_set_epi8(255,255,255,255,255,255,255,255,
			      7 ,6 ,5, 4, 3, 2, 1, 0);
    __m128i mask_second = _mm_set_epi8(7 ,6 ,5, 4, 3, 2, 1, 0,
			      255,255,255,255,255,255,255,255);

    running_sum1 = (__m128d)_mm_shuffle_epi8((__m128i)running_sum1, mask_first);
    running_sum2 = (__m128d)_mm_shuffle_epi8((__m128i)running_sum2, mask_second);
    running_sum1 = (__m128d)_mm_or_si128((__m128i)running_sum1, (__m128i)running_sum2);

    running_sum3 = (__m128d)_mm_shuffle_epi8((__m128i)running_sum3, mask_first);
    running_sum4 = (__m128d)_mm_shuffle_epi8((__m128i)running_sum4, mask_second);
    running_sum2 = (__m128d)_mm_or_si128((__m128i)running_sum3, (__m128i)running_sum4);

    running_sum5 = (__m128d)_mm_shuffle_epi8((__m128i)running_sum5, mask_first);
    running_sum6 = (__m128d)_mm_shuffle_epi8((__m128i)running_sum6, mask_second);
    running_sum3 = (__m128d)_mm_or_si128((__m128i)running_sum6, (__m128i)running_sum5);

    running_sum7 = (__m128d)_mm_shuffle_epi8((__m128i)running_sum7, mask_first);
    running_sum8 = (__m128d)_mm_shuffle_epi8((__m128i)running_sum8, mask_second);
    running_sum4 = (__m128d)_mm_or_si128((__m128i)running_sum8, (__m128i)running_sum7);

    // RS 1-4 are right and expected here too
    // rs 5-8 neglected and not required from now

    __m128i a01_round = convert_double_to_f48_SSE((__m128i)running_sum1);
    __m128i a23_round = convert_double_to_f48_SSE((__m128i)running_sum2);
    __m128i a45_round = convert_double_to_f48_SSE((__m128i)running_sum3);
    __m128i a67_round = convert_double_to_f48_SSE((__m128i)running_sum4);

    // place them right for memory write
    __m128i match_mask = _mm_set_epi8(3,2,1,0,255,255,255,255,255,255,255,255,255,255,255,255); // mask used to match the missing spaces
    __m128i a23_shuffled = _mm_shuffle_epi8((__m128i)a23_round, match_mask); // shuffle the positions required for the space in a01 for a2
    a01_round = _mm_or_si128(a01_round,a23_shuffled);

    a23_round = _mm_srli_si128 (a23_round, 4); // using _mm_srli_si128 instead of _mm_sll_epi64 because the epi64 shitfs witin each double element in the 128 item

    match_mask = _mm_set_epi8(7,6,5,4,3,2,1,0,255,255,255,255,255,255,255,255); // reset the match mask for a4 and small bit of a5
    __m128i a45_shuffled = _mm_shuffle_epi8((__m128i)a45_round, match_mask); // shuffle a45 to fit in a23
    a23_round = _mm_or_si128(a23_round,a45_shuffled);

    a45_round = _mm_srli_si128(a45_round, 8); // using _mm_srli_si128 instead of _mm_sll_epi64 because the epi64 shitfs witin each double element in the 128 item

    match_mask = _mm_set_epi8(11,10,9,8,7,6,5,4,3,2,1,0,255,255,255,255);
    __m128i a67_shuffled = _mm_shuffle_epi8((__m128i)a67_round, match_mask);
    a45_round = _mm_or_si128(a45_round,a67_shuffled);
     // WRITE BACK TO MEMORY!
    _mm_storeu_pd((double*)&result[i], (__m128d)a01_round);
    _mm_storeu_pd(bofs(&result[i],2), (__m128d)a23_round);
    _mm_storeu_pd(bofs(&result[i],4), (__m128d)a45_round);
  }
  vec = result;
}
    /*** END LEVEL 2 ***/
