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
    /*** LEVEL 1 ***/

void scale_f48_vector_SSE (f48 * a, f48 scalar)
{
   double double_scalar = (double)scalar;
  __m128d scalar_vec = _mm_load1_pd(&double_scalar);
  // set the mask to be used by loading in the unrolled loop
  __m128i mask = _mm_set_epi8(11, 10, 9, 8,  7,  6, 255, 255,
              5, 4, 3, 2, 1, 0, 255, 255);


  for ( int i = 0; i < size; i+=8 ) { // should be size

    // try loading all in 3 SSE items, then shuffle them into 4 128 items and then do the scale and then the conversion and reshuffling in place and storing!!!
    // this will give another set of results for comparison of the methods with the benchmark being the double version

    // ^^ for loop unrolling to work need to loop around 8 elements
    __m128i a01 = _mm_loadu_si128((__m128i*)(&a[i])); // load a[0] and a[1]
    a01 = _mm_shuffle_epi8(a01, mask);

    __m128i a23 = _mm_loadu_si128((__m128i*)(&a[i+2])); // load a[2] and a[3]
    a23 = _mm_shuffle_epi8(a23, mask);

    __m128i a45 = _mm_loadu_si128((__m128i*)(&a[i+4])); // load a[4] and a[5]
    a45 = _mm_shuffle_epi8(a45, mask);

    __m128i a67 = _mm_loadu_si128((__m128i*)(&a[i+6])); // load a[6] and a[7]
    a67 = _mm_shuffle_epi8(a67, mask);

    // SCALE THEM UP
    __m128d res_a01 = _mm_mul_pd((__m128d)a01, (__m128d)scalar_vec);
    __m128d res_a23 = _mm_mul_pd((__m128d)a23, (__m128d)scalar_vec);
    __m128d res_a45 = _mm_mul_pd((__m128d)a45, (__m128d)scalar_vec);
    __m128d res_a67 = _mm_mul_pd((__m128d)a67, (__m128d)scalar_vec);

    //ROUND THEM!!!
    __m128i a01_round = convert_double_to_f48_SSE((__m128i)res_a01);
    __m128i a23_round = convert_double_to_f48_SSE((__m128i)res_a23);
    __m128i a45_round = convert_double_to_f48_SSE((__m128i)res_a45);
    __m128i a67_round = convert_double_to_f48_SSE((__m128i)res_a67);

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

    #define bofs(base, ofs) (((double*)(base))+ofs) // INTERESTING TO TALK ABOUT THIS LITTLE FUNCTION!!!

    // WRITE BACK TO MEMORY
    _mm_storeu_pd(bofs(&a[i],0), (__m128d)a01_round);
    _mm_storeu_pd(bofs(&a[i],2), (__m128d)a23_round);
    _mm_storeu_pd(bofs(&a[i],4), (__m128d)a45_round);
  }
}

void scale_double_vector_SSE (double * a, double scalar)
{
  __m128d scalar_vec = _mm_load1_pd(&scalar);
  __m128d result_vec = _mm_set1_pd(0.0);

  for (int i=0; i<size; i+=2) { // should be size
    __m128d a_vec = _mm_load_pd(&a[i]);
    result_vec = _mm_mul_pd(a_vec, scalar_vec);
    _mm_store_pd(&a[i], result_vec);
  }
}

double dot_product_SSE_double (double *a, double *b) {
	double total=0;
	__m128d result_vec = _mm_set1_pd(0.0); // result initially 0 - running sum
	__m128d temp_vect;

	for ( int i = 0; i < size; i+= 2 ) {
		// load vectors
		__m128d a_vec = _mm_load_pd(&a[i]);
		__m128d b_vec = _mm_load_pd(&b[i]);
		// a & b vectors loaded
		// compute multiplication and save temporary = a[1]*b[1]   a[0]*b[0]
		 temp_vect = _mm_mul_pd(a_vec, b_vec);
		 result_vec = _mm_add_pd(temp_vect, result_vec);  //performs vertical addition
	}
  __m128i mask_hadd = _mm_set_epi8(7 ,6 ,5, 4, 3, 2, 1, 0,
                        15, 14, 13, 12, 11, 10, 9, 8);
  __m128i result_shuffled = _mm_shuffle_epi8((__m128i)result_vec, mask_hadd);
	result_vec = _mm_add_pd((__m128d)result_shuffled, result_vec); // cumulate result
	// store result into double
// 	_mm_store1_pd(&total, result_vec);
	_mm_store_sd(&total, result_vec);
	return total;
}

f48 dot_product_SSE_f48 (f48 *a, f48 *b){
	double total=0;
	__m128d result_vec = _mm_set1_pd(0.0); // result initially 0 - running sum
	__m128i mask = _mm_set_epi8(11, 10, 9, 8, 7, 6, 255, 255,
					5, 4, 3, 2, 1, 0, 255, 255);

	for ( int i = 0; i < size; i+= 2 ) {
		__m128i a_vec = _mm_loadu_si128((__m128i*)(&a[i]));
		a_vec = _mm_shuffle_epi8(a_vec, mask);
		__m128i b_vec = _mm_loadu_si128((__m128i*)(&b[i]));
		b_vec = _mm_shuffle_epi8(b_vec, mask);
		a_vec = (__m128i)_mm_mul_pd((__m128d)a_vec, (__m128d)b_vec);
		result_vec = _mm_add_pd((__m128d)a_vec, result_vec);  //performs vertical addition
	}
  __m128i mask_hadd = _mm_set_epi8(7 ,6 ,5, 4, 3, 2, 1, 0,
          15, 14, 13, 12, 11, 10, 9, 8);
  __m128i result_shuffled = _mm_shuffle_epi8((__m128i)result_vec, mask_hadd);
	result_vec = _mm_add_pd((__m128d)result_shuffled, result_vec); // cumulate result
  //TODO: instead of hadd try permute and add!!!

// 	_mm_store1_pd(&total, result_vec);
	_mm_store_sd(&total, result_vec);
	f48 total_result (total);
	return total_result;
}

f48 absolute_max_SSE_f48 (f48 *a){
  __m128i mask = _mm_set_epi8(11, 10, 9, 8, 7, 6, 255, 255,
			       5,  4, 3, 2, 1, 0, 255, 255);
  // load the first two as being the max
  __m128i max = _mm_loadu_si128((__m128i*)(&a[0]));
  max = _mm_shuffle_epi8(max, mask);
  for ( int i = 2; i < size; i+= 2 ) { // start loop from the third element two by two until the end
    __m128i a_vec = _mm_loadu_si128((__m128i*)(&a[i]));
    a_vec = _mm_shuffle_epi8(a_vec, mask);
    max = (__m128i)_mm_max_pd((__m128d)max, (__m128d)a_vec);
  }
  // at the end compare max with suffle max and find the maximum value that needs to be stored in a double that requires conversion back to f48
  mask = _mm_set_epi8(7 ,6 ,5, 4, 3, 2, 1, 0,
		      15, 14, 13, 12, 11, 10, 9, 8);
  __m128i max_shuffled = _mm_shuffle_epi8(max, mask);
  __m128d max_result = _mm_max_pd((__m128d)max, (__m128d)max_shuffled);
  // store one of max into a double and print double
  double maximum=0;
//   _mm_store1_pd(&maximum, max_result);
  _mm_store_sd(&maximum, max_result);
  f48 max_f48 (maximum);
  return max_f48;
}

f48 absolute_min_SSE_f48 (f48 *a) {
    __m128i mask = _mm_set_epi8(11, 10, 9, 8, 7, 6, 255, 255,
			       5,  4, 3, 2, 1, 0, 255, 255);
  // load the first two as being the min
  __m128i min = _mm_loadu_si128((__m128i*)(&a[0]));
  min = _mm_shuffle_epi8(min, mask);
  for ( int i = 2; i < size; i+= 2 ) { // start loop from the third element two by two until the end
    __m128i a_vec = _mm_loadu_si128((__m128i*)(&a[i]));
    a_vec = _mm_shuffle_epi8(a_vec, mask);
    min = (__m128i)_mm_min_pd((__m128d)min, (__m128d)a_vec);
  }
  // at the end compare min with suffle min and find the maximum value that needs to be stored in a double that requires conversion back to f48
  mask = _mm_set_epi8(7 ,6 ,5, 4, 3, 2, 1, 0,
		      15, 14, 13, 12, 11, 10, 9, 8);
  __m128i min_shuffled = _mm_shuffle_epi8(min, mask);
  __m128d min_result = _mm_min_pd((__m128d)min, (__m128d)min_shuffled);
  // store one of min into a double and print double
  double minimum=0;
//   _mm_store1_pd(&minimum, min_result);
  _mm_store_sd(&minimum, min_result);
  f48 max_f48 (minimum);
  return max_f48;
}

double absolute_max_SSE_double (double *a){
  __m128d max = _mm_load_pd(&a[0]);
  for ( int i = 2; i < size; i+= 2 ) {
    __m128d a_vec = _mm_load_pd(&a[i]);
    max = _mm_max_pd(max, a_vec);
  }
  __m128i mask = _mm_set_epi8(7 ,6 ,5, 4, 3, 2, 1, 0,
		      15, 14, 13, 12, 11, 10, 9, 8);
  __m128i max_shuffled = _mm_shuffle_epi8((__m128i)max, mask);
  __m128d max_result = _mm_max_pd(max, (__m128d)max_shuffled);
  double maximum=0;
//   _mm_store1_pd(&maximum, max_result);
  _mm_store_sd(&maximum, max_result);
  return maximum;
}

double absolute_min_SSE_double (double *a){
  __m128d min = _mm_load_pd(&a[0]);
  for ( int i = 2; i < size; i+= 2 ) {
    __m128d a_vec = _mm_load_pd(&a[i]);
    min = _mm_min_pd(min, a_vec);
  }
  __m128i mask = _mm_set_epi8(7 ,6 ,5, 4, 3, 2, 1, 0,
		      15, 14, 13, 12, 11, 10, 9, 8);
  __m128i min_shuffled = _mm_shuffle_epi8((__m128i)min, mask);
  __m128d min_result = _mm_min_pd(min, (__m128d)min_shuffled);
  double minimum=0;
//   _mm_store1_pd(&minimum, min_result);
  _mm_store_sd(&minimum, min_result);
  return minimum;
}

f48 magnitude_SSE_f48 (f48 *a){
  __m128d result_vec = _mm_set1_pd(0.0); // result initially 0 - running sum
  __m128i mask = _mm_set_epi8(11, 10, 9, 8, 7, 6, 255, 255,
			       5,  4, 3, 2, 1, 0, 255, 255);
  for(int i=0; i<size; i+=2){
   __m128i a_vect = _mm_loadu_si128((__m128i*)(&a[i]));
   a_vect = _mm_shuffle_epi8(a_vect, mask);
   a_vect = (__m128i)_mm_mul_pd((__m128d)a_vect,(__m128d)a_vect); // ^2
   result_vec = _mm_add_pd(result_vec, (__m128d)a_vect); // running sum
  }
  __m128i mask_hadd = _mm_set_epi8(7 ,6 ,5, 4, 3, 2, 1, 0,
          15, 14, 13, 12, 11, 10, 9, 8);
  __m128i result_shuffled = _mm_shuffle_epi8((__m128i)result_vec, mask_hadd);
  result_vec = _mm_add_pd((__m128d)result_shuffled, result_vec);
  result_vec = _mm_sqrt_pd(result_vec);
  double res=0;
//   _mm_store1_pd(&res, result_vec);
  _mm_store_sd(&res, result_vec);
  return f48(res);
}

double magnitude_SSE_double (double *a){
  __m128d result_vec = _mm_set1_pd(0.0); // result initially 0 - running sum
  for ( int i = 0; i < size; i+= 2 ) {
    __m128d a_vect = _mm_load_pd(&a[i]);
    a_vect = _mm_mul_pd(a_vect, a_vect); // ^2
    result_vec = _mm_add_pd(result_vec, a_vect); // running sum
  }
  __m128i mask_hadd = _mm_set_epi8(7 ,6 ,5, 4, 3, 2, 1, 0,
          15, 14, 13, 12, 11, 10, 9, 8);
  __m128i result_shuffled = _mm_shuffle_epi8((__m128i)result_vec, mask_hadd);
  result_vec = _mm_add_pd((__m128d)result_shuffled, result_vec);
  result_vec = _mm_sqrt_pd(result_vec);
  double res=0;
//   _mm_store1_pd(&res, result_vec);
  _mm_store_sd(&res, result_vec);
  return res;
}
    /*** END LEVEL 1 ***/
