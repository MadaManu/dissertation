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

using namespace std;

struct stat st = {0};

int size = 1024*1024;
int runs = 100;

typedef unsigned long long u64;
typedef unsigned short u16;

union conversion_union {
  unsigned long long l;
  double d;
  float f;
} convert;

/************** class u48 **********************************/
class u48 {
private:
  unsigned long long num:48;
  
public:
  u48() { };
  u48(u64 value) { num = value; };
  operator u64() { return num; };
} __attribute__((packed));

/************* class f48 ************************************/
class f48 {
private:
  unsigned long long num:48;  
public:
  f48() { };
  f48(double value);
  operator double();
} __attribute__((packed));


/*************** F48 ***************/
f48::f48(double value)
{
  // TODO: optimize this code!
  // convert to 64-bit pattern
//   u64 tmp = _castf64_u64(value);// not valid INTEL operation
  convert.d = value;
  unsigned long long tmp = convert.l;
  // round to nearest even is a little complex
  // the u64 number has the following format:
  // 47upper_bits:L:G:15lower_bits
  
  bool S = (tmp & 65535)>0 ? 1:0; // STICKY bit - OR of the remaining bits
  bool R = (tmp >> 15) & 1;  // ROUND bit - first bit removed
  bool G = (tmp >> 16) & 1;  // GUARD bit - LSB of result
  u64 mantissa = (tmp&4503599627370495)>>16; // extracts the remainding mantissa from number
  unsigned long long signExponent = (tmp>>52)<<52; // clear the mantissa
/*
 * Rounding to the nearest even
 * 
 * G = 0 - do nothing just truncate
 * G = 1 - check R&S as below
 * 
 * R  S
 * 0  0 - TIE (check bit before G: if 1 up else nothing
 * 0  1 - up
 * 1  0 - up
 * 1  1 - up
 */
 
  if(G){
	  if (R||S){ // R or S is true go up
		  mantissa = mantissa +1;
		  if((mantissa>>36)&1){ // overflow of mantissa
			  mantissa = 0; // reset mantissa
			  // add one to exponent
			  // extract exponent only add one to it and check for overflow
			  unsigned long long exponent_mask = (unsigned long long)(1)<<63;
			  exponent_mask = ~exponent_mask;
			  unsigned long long exponent = (signExponent & exponent_mask);
			  exponent += 1;
			  exponent = exponent >> 12;
			  if (exponent>0) { // overflow on exponent set to +/- infinity
				  signExponent = signExponent | ((unsigned long long)2047<<52); // set all the bits of the exponent to 1 keeping sign
				  mantissa = mantissa & 0; // make sure mantissa is 0
			  }
		  }
	  } else { // TIE situation
	  // add 1 if the mantissa is odd
	  // add 1 if the mantissa is even
		  // check bit before G
		  if((mantissa>>1)&1) {
			  mantissa = mantissa +1;
			  if((mantissa>>36)&1){ // overflow of mantissa
				  mantissa = 0; // reset mantissa
				  // add one to exponent
				  // extract exponent only add one to it and check for overflow
				   unsigned long long exponent_mask = (unsigned long long)(1)<<63;
				  exponent_mask = ~exponent_mask;
				  unsigned long long exponent = (signExponent & exponent_mask);
				  exponent += 1;
				  exponent = exponent >> 12;
				  if (exponent>0) { // overflow on exponent set to +/- infinity
					  signExponent = signExponent | ((unsigned long long)2047<<52); // set all the bits of the exponent to 1 keeping sign
					  mantissa = mantissa & 0; // make sure mantissa is 0
				  }
			  }
		  } // else do nothing 
	  }
  } else {
	  // not required as if G is 0 R&S are x (don't cares)
  }
  unsigned long long result = (mantissa<<16) + signExponent;
// to convert the number back to double there is a need of having the conversion 
// done in the union (convert)
convert.l = result;
// cout<<convert.d;
	// compensate for little endianess
  this->num = result >> 16;
  
}

f48::operator double()
{
  // convert to double
  convert.l = (this->num)<<16; // pad to compensate for little endianess
  return convert.d; 
}

/****************** HELPER FUNCTIONS ******************/

template <class T>
void populate_array(T * a)
{
  //srand((unsigned)time(0)); 
  srand(5);

  for ( int i = 0; i < size; i++ ) {
    a[i] = rand() % 1024;
  }
}

template <class T>
void populate_matrix(T ** a)
{
  //srand((unsigned)time(0)); 
  srand(5);
  for(int i = 0; i < size; ++i){
      a[i] = new T[size];
  }
  for ( int i = 0; i < size; i++ ) {
    for (int j=0; j<size; j++) {
	a[i][j] = rand() % 1024;
    }    
  }
}

unsigned long long rdtsc()
{
  #define rdtsc_macro(low, high) \
         __asm__ __volatile__("rdtsc" : "=a" (low), "=d" (high))

  unsigned int low, high;
  rdtsc_macro(low, high);
  return (((u64)high) << 32) | low;
}



/************** CONVERSION FUNCTION SSE **************/
__m128i convert_double_to_f48_SSE (__m128i a)
{
	// convert from 2 double in vector to 2 rounded f48  
	__m128i mask = _mm_set_epi8(15,14,13,12,11,10,255, 255,
			    7, 6, 5, 4, 3, 2, 255, 255);
	__m128i unrounded_result = _mm_shuffle_epi8(a, mask);
	
	__m128d s_mask = _mm_set_pd(1.61890490173e-319,1.61890490173e-319);  // 0000000000000000000000000000000000000000000000000111111111111111
							     // 0x7fff ^^ - to be added for S
	__m128i s = _mm_and_si128(a, (__m128i)s_mask); // remove all the other stuff before S and extract S bits (last bits after R being removed)
	s = (__m128i)_mm_add_pd((__m128d)s, (__m128d)s_mask); // add 0x7fff to obtain S in overflow at position of R
	
	__m128d r_mask = _mm_set_pd(8.09477154146e-320,8.09477154146e-320); // 0000000000000000000000000000000000000000000000000100000000000000
							      // 0x4000 && - to be used to select R
	__m128i r = _mm_and_si128(a, (__m128i)r_mask); // select the R bit
	
	__m128i r_or_s = _mm_or_si128(s,r);  // R|S in the position of R
		
	__m128d shift_count = _mm_set1_pd(1.0);
	r_or_s = _mm_sll_epi64(r_or_s, (__m128i)shift_count); // shift R|S to left by 1 to match the position of G
	
	__m128d result = _mm_add_pd((__m128d)unrounded_result, (__m128d)r_or_s);
	
	// permute to left
	__m128i permute_mask = _mm_set_epi8(255,255,255,255,15, 14, 13, 12, 11, 10, 7, 6,
			    5, 4, 3, 2);
	
	return (__m128i)_mm_shuffle_epi8((__m128i)result, permute_mask); // apply permutation mask
// 	return (__m128i)result;
}



/************ IMPLEMENTATION ************/

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
	result_vec = _mm_hadd_pd(result_vec, result_vec); // cumulate result
	// store result into double
// 	_mm_store1_pd(&total, result_vec);
	_mm_store_sd(&total, result_vec);
	return total;
}

f48 dot_product_SSE_f48 (f48 *a, f48 *b){
	double total=0;
	__m128d result_vec = _mm_set1_pd(0.0); // result initially 0 - running sum
	__m128d temp_vect;
	__m128i mask = _mm_set_epi8(11, 10, 9, 8, 7, 6, 255, 255,
					5, 4, 3, 2, 1, 0, 255, 255);

	for ( int i = 0; i < size; i+= 2 ) {
		__m128i a_vec = _mm_loadu_si128((__m128i*)(&a[i]));
		a_vec = _mm_shuffle_epi8(a_vec, mask);
		__m128i b_vec = _mm_loadu_si128((__m128i*)(&b[i]));
		b_vec = _mm_shuffle_epi8(b_vec, mask);
		 temp_vect = _mm_mul_pd((__m128d)a_vec, (__m128d)b_vec);
		 result_vec = _mm_add_pd(temp_vect, result_vec);  //performs vertical addition
	}
	result_vec = _mm_hadd_pd(result_vec, result_vec); // cumulate result
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
  result_vec = _mm_hadd_pd(result_vec, result_vec);
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
  result_vec = _mm_hadd_pd(result_vec, result_vec);
  result_vec = _mm_sqrt_pd(result_vec);
  double res=0;
//   _mm_store1_pd(&res, result_vec);
  _mm_store_sd(&res, result_vec);
  return res;
}
    /*** END LEVEL 1 ***/

    /*** LEVEL 2 ***/
// TODO: implement same using f48 maybe to compare some timmings
// TODO: function to take size
// function is void as should overwrite the input
// works for square matrix - ?? does it require to work for not square matrix?
// ^^ for non-square matrix requires computation of final size of vector
// matrix * vector always returns vector (size is dependant on the matrix size
void matrix_vector_mul_double(double** mat, double* &vec)
{
  double* result = new double[size];
  for(unsigned i=0;i<size;i++) { // row
    double running_sum = 0;
    for(unsigned j=0;j<size;j++) { // col
	running_sum += mat[i][j]*vec[j];
    }
    result[i] = running_sum;
  }
  vec = result;
}

void matrix_vector_mul_f48(f48** mat, f48* &vec) 
{
  f48* result = new f48[size];
  for(unsigned i=0;i<size;i++) { // row
    double running_sum = 0;
    for(unsigned j=0;j<size;j++) { // col
	running_sum += double(mat[i][j])*double(vec[j]);
    }
    result[i] = (f48)running_sum;
  }
  vec = result;
}

// TODO: function to take size
// TODO: requires implementation
// matrix is square?
// same questions as above
void matrix_vector_mul_SSE_double(double** mat, double* &vec)
{
  double* result = new double[size]; // should be size of result!
  for(unsigned i=0;i<size;i++) { // row
    __m128d running_sum = _mm_set1_pd(0.0); // running sum initially 0
    for(unsigned j=0;j<size;j+=2) { // col - requires skipping on 2 at a time
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
  f48* result = new f48[size]; // should be size of result!
  __m128i mask = _mm_set_epi8(11, 10, 9, 8, 7, 6, 255, 255,
  			      5, 4, 3, 2, 1, 0, 255, 255);
  __m128i shuffling_mask = _mm_set_epi8(7 ,6 ,5, 4, 3, 2, 1, 0,
			      15, 14, 13, 12, 11, 10, 9, 8);
  for(unsigned i=0;i<size;i++) { // row
    __m128d running_sum = _mm_set1_pd(0.0); // running sum initially 0
    for(unsigned j=0;j<size;j+=2) { // col - requires skipping on 2 at a time

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
  double* result = new double[size];
  for(unsigned i=0;i<size;i+=2) { // row // requiring 2 at a time
    __m128d running_sum1 = _mm_set1_pd(0.0); // running sum initially 0
    __m128d running_sum2 = _mm_set1_pd(0.0); // running sum initially 0
    for(unsigned j=0;j<size;j+=2) { // col - requires skipping on 2 at a time
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
  f48* result = new f48[size];
  __m128i load_mask = _mm_set_epi8(11, 10, 9, 8, 7, 6, 255, 255,
  			      5, 4, 3, 2, 1, 0, 255, 255);
  for(unsigned i=0;i<size;i+=2) { // row // requiring 2 at a time
    __m128d running_sum1 = _mm_set1_pd(0.0); // running sum initially 0
    __m128d running_sum2 = _mm_set1_pd(0.0); // running sum initially 0
    for(unsigned j=0;j<size;j+=2) { // col - requires skipping on 2 at a time
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
    // TESTING change size to min 8 - but multiple of 8
    f48* result = new f48[size];
  __m128i load_mask = _mm_set_epi8(11, 10, 9, 8, 7, 6, 255, 255,
  			      5, 4, 3, 2, 1, 0, 255, 255);
  for(unsigned i=0;i<size;i+=8) { // row // requiring 8 at a time - because loop un-roll
    __m128d running_sum1 = _mm_set1_pd(0.0); // running sum initially 0
    __m128d running_sum2 = _mm_set1_pd(0.0); // running sum initially 0
    __m128d running_sum3 = _mm_set1_pd(0.0); // running sum initially 0
    __m128d running_sum4 = _mm_set1_pd(0.0); // running sum initially 0
    __m128d running_sum5 = _mm_set1_pd(0.0); // running sum initially 0
    __m128d running_sum6 = _mm_set1_pd(0.0); // running sum initially 0
    __m128d running_sum7 = _mm_set1_pd(0.0); // running sum initially 0
    __m128d running_sum8 = _mm_set1_pd(0.0); // running sum initially 0
    
    for(unsigned j=0;j<size;j+=2) { // col - requires skipping on 2 at a time
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

    /*** LEVEL 3 ***/
// TODO ADD L3 implementation

      /** MATRIX MATRIX MUL double **/
double** matrix_matrix_mul_double(double** a, double** b) 
{
  double** res = new double*[size];
  for(int i=0; i<size; i++) {
    res[i] = new double[size];
  }
  for(int i=0; i<size; i++) { // a row count
    for(int j=0; j<size; j++) { // b column count
      double sum = 0; // init sum
      for(int k=0; k<size; k++) { // k offset
        sum += a[i][k]*b[k][j];
      }
      res[i][j] = sum;
    }
  }
  return res;
}

      /** MATRIX MATRIX MUL f48 **/
f48** matrix_matrix_mul_f48(f48** a, f48** b) {
  f48** res = new f48*[size];
  for(int i=0; i<size; i++) {
    res[i] = new f48[size];
  }
  for(int i=0; i<size; i++) {
    for(int j=0; j<size; j++) {
      double sum = 0; // compute the sum in doubles
      for(int k=0; k<size; k++) {
        sum += double(a[i][k])*double(b[k][j]);
      }
      // save back to memory in f48 format
      res[i][j] = f48(sum);
    }
  }
  return res;
}

      /** MATRIX MATRIX MUL double SSE **/
double** matrix_matrix_mul_double_SSE() {
  double** res = new double*[8];
  for(int i=0; i<8; i++) {
    res[i] = new double[8];
  }

  // do magic here

  return res;
}

    /*** END LEVEL 3 ***/

/************ END IMPLEMENTATION ************/

/************ TESTS ************/

    /*** LEVEL 1 ***/

      /** DOT PRODUCT double **/
void test_double_dot_prod()
{
  cout<<"DOUBLE(SSE) dot product..." << endl;
  // declaration
  double * a = new double[size];
  double * b = new double[size];
  u64 start,stop,diff;
  ofstream myfile;

  // initialisation
  populate_array(a);
  populate_array(b);
  char path[256];
  sprintf(path, "results/level1/%d/dot_prod_SSE_double.txt", size);
  myfile.open (path);

  // run tests
  for (int i=0; i<runs; i++) {
    start = rdtsc();
    dot_product_SSE_double(a, b);
    stop = rdtsc();
    diff = stop - start;
    myfile<<diff<<endl;
    // repopulate
    populate_array(a);
    populate_array(b);
  }
  myfile.close();
  cout<<"Done. Results in results/level1/"<<size<<"/dot_prod_SSE_double.txt"<<endl;
}

      /** DOT PRODUCT f48 **/
void test_f48_dot_prod()
{
  cout<<"F48(SSE) dot product..." << endl;

  // declaration
  f48 * a = new f48[size];
  f48 * b = new f48[size];
  u64 start,stop,diff;
  ofstream myfile;
  
  // initialisation
  populate_array(a);
  populate_array(b);
  char path[256];
  sprintf(path, "results/level1/%d/dot_prod_SSE_f48.txt", size);
  myfile.open (path);

  // run tests
  for(int i=0; i<runs; i++) {
    start = rdtsc();
    dot_product_SSE_f48(a, b);
    stop = rdtsc();
    diff = stop - start;
    myfile<<diff<<endl;
    // repopulate
    populate_array(a);
    populate_array(b);
  }
  myfile.close();
  cout<<"Done. Results in results/level1/"<<size<<"/dot_prod_SSE_f48.txt"<<endl;
}

      /** SCALE double **/
void test_double_scale()
{
  cout<<"DOUBLE(SSE) scale..." << endl;
  // declaration
  double * a = new double[size];
  double scalar;
  u64 start,stop,diff;
  ofstream myfile;
  
  // initialisation
  srand(5);
  scalar = rand() % 1024;
  populate_array(a);
  char path[256];
  sprintf(path, "results/level1/%d/scale_SSE_double.txt", size);
  myfile.open (path);

  // run tests
  for(int i=0; i<runs; i++) {
    start = rdtsc();
    scale_double_vector_SSE(a, scalar);
    stop = rdtsc();
    diff = stop - start;
    myfile<<diff<<endl;
    // repopulate
    srand(5);
    scalar = rand() % 1024;
    populate_array(a);
  }
  myfile.close();
  cout<<"Done. Results in results/level1/"<<size<<"/scale_SSE_double.txt"<<endl;
}

      /** SCALE f48 **/
void test_f48_scale()
{
  cout<<"F48(SSE) scaling SSE..." << endl;

  // declaration
  f48 * a = new f48[size];
  f48 scalar;
  u64 start, stop, diff;
  ofstream myfile;
  
  // initialisation
  populate_array(a);
  srand(5);
  scalar = f48(rand() % 1024);
  char path[256];
  sprintf(path, "results/level1/%d/scale_SSE_f48.txt", size);
  myfile.open (path);

  // run tests
  for(int i=0; i<runs; i++) {
    start = rdtsc();
    scale_f48_vector_SSE(a, scalar);
    stop = rdtsc();
    diff = stop - start;
    myfile<<diff<<endl;
    // repopulate
    populate_array(a);
    srand(5);
    scalar = f48(rand() % 1024);
  }
  myfile.close();
  cout<<"Done. Results in results/level1/"<<size<<"/scale_SSE_f48.txt"<<endl;
}

      /** MAX double **/
void test_max_double_SSE()
{
  cout<<"DOUBLE(SSE) absolute maximum..."<<endl;

  // declaration
  double * a = new double[size];
  u64 start, stop, diff;
  ofstream myfile;

  // initialisation
  populate_array(a);
  char path[256];
  sprintf(path, "results/level1/%d/max_SSE_double.txt", size);
  myfile.open (path);

  // run tests
  for(int i=0; i<runs; i++) {
    start = rdtsc();
    absolute_max_SSE_double(a);
    stop = rdtsc();
    diff = stop - start;
    myfile<<diff<<endl;
    // repopulate
    populate_array(a);
  }
  myfile.close();
  cout<<"Done. Results in results/level1/"<<size<<"/max_SSE_double.txt"<<endl;
}

      /** MAX f48 **/
void test_max_f48_SSE()
{
  cout<<"F48(SSE) absolute maximum..."<<endl;
  // declaration
  f48 * a = new f48[size];
  u64 start, stop, diff;
  ofstream myfile;

  // initialisation
  populate_array(a);
char path[256];
  sprintf(path, "results/level1/%d/max_SSE_f48.txt", size);
  myfile.open (path);

  // run tests
  for(int i=0; i<runs; i++) {
    start = rdtsc();
    absolute_max_SSE_f48(a);
    stop = rdtsc();
    diff = stop - start;
    myfile<<diff<<endl;
    // repopulate
    populate_array(a);
  }
  myfile.close();
  cout<<"Done. Results in results/level1/"<<size<<"/max_SSE_f48.txt"<<endl;
}

      /** MIN double **/
void test_min_double_SSE()
{
  cout<<"DOUBLE(SSE) absolute minimum..."<<endl;
  // declaration
  double * a = new double[size];
  u64 start, stop, diff;
  ofstream myfile;

  // initialisation
  populate_array(a);
char path[256];
  sprintf(path, "results/level1/%d/min_SSE_double.txt", size);
  myfile.open (path);

  // run tests
  for(int i=0; i<runs; i++) {
    start = rdtsc();
    absolute_min_SSE_double(a);
    stop = rdtsc();
    diff = stop - start;
    myfile<<diff<<endl;
    // repopulate
    populate_array(a);
  }
  myfile.close();
  cout<<"Done. Results in results/level1/"<<size<<"/min_SSE_double.txt"<<endl;
}

      /** MIN f48 **/
void test_min_f48_SSE()
{
  cout<<"F48(SSE) absolute minimum..."<<endl;
  // declaration
  f48 * a = new f48[size];
  u64 start, stop, diff;
  ofstream myfile;

  // initialisation
  populate_array(a);
char path[256];
  sprintf(path, "results/level1/%d/min_SSE_f48.txt", size);
  myfile.open (path);

  // run tests
  for(int i=0; i<runs; i++) {
    start = rdtsc();
    absolute_min_SSE_f48(a);
    stop = rdtsc();
    diff = stop - start;
    myfile<<diff<<endl;
    // repopulate
    populate_array(a);
  }
  myfile.close();
  cout<<"Done. Results in results/level1/"<<size<<"/min_SSE_f48.txt"<<endl;
}

void test_magnitude_double_SSE()
{
  cout<<"DOUBLE(SSE) magnitude..." << endl;
  // declaration
  double * a = new double[size];
  u64 start, stop, diff;
  ofstream myfile;
  
  // initialisation
  populate_array(a);
  char path[256];
  sprintf(path, "results/level1/%d/magnitude_SSE_double.txt", size);
  myfile.open (path);

  // run tests
  for(int i=0; i<runs; i++) {
    start = rdtsc();
    magnitude_SSE_double(a);
    stop = rdtsc();
    diff = stop - start;
    myfile <<diff<<endl;
    // repopulate
    populate_array(a);
  }
  myfile.close();
  cout<<"Done. Results in results/level1/"<<size<<"/magnitude_SSE_double.txt"<<endl;
}

      /** MAGNITUDE f48 **/
void test_magnitude_f48_SSE()
{
  cout<<"F48(SSE) magnitude..." << endl;
  // declaration
  f48 * a = new f48[size];
  u64 start, stop, diff;
  ofstream myfile;

  // initialisation
  populate_array(a);
  char path[256];
  sprintf(path, "results/level1/%d/magnitude_SSE_f48.txt", size);
  myfile.open (path);

  // run tests  
  for(int i=0; i<runs; i++) {
    start = rdtsc();
    magnitude_SSE_f48(a);
    stop = rdtsc();
    diff = stop - start;
    myfile<<diff<<endl;
    // repopulate
    populate_array(a);
  }
  myfile.close();
  cout<<"Done. Results in results/level1/"<<size<<"/magnitude_SSE_f48.txt"<<endl;
}
    /*** END LEVEL ***/


    /*** LEVEL 2 ***/

      /** MATRIX VECTOR MUL double **/
void test_matrix_vector_mul_double_nonSSE()
{
  cout<<"DOUBLE(NON-SSE) matrix-vector multiplication..." << endl;
  double* a = new double[size];
  double** matrix = new double*[size];
  u64 start, stop, diff;

  populate_matrix(matrix);
  populate_array(a);

  // preparing file
  ofstream outputfile;
  char path[256];
  sprintf(path, "results/level2/%d/matrix_vector_mul_double_nonSSE.txt", size);
  outputfile.open (path);

  for(int i=0; i<runs; i++) {
    start = rdtsc();
    matrix_vector_mul_double(matrix,a);
    stop = rdtsc();
    diff = stop - start;
    outputfile<<diff<<endl;
    // repopulate 
    populate_matrix(matrix);
    populate_array(a);
  }
  outputfile.close();
  cout<<"Done. Results in results/level2/"<<size<<"/matrix_vector_mul_double_nonSSE.txt"<<endl;
}

void test_matrix_vector_mul_f48_nonSSE()
{
    cout<<"F48(NON-SSE) matrix-vector multiplication..." << endl;
  f48* a = new f48[size];
  f48** matrix = new f48*[size];
  populate_matrix(matrix);
  populate_array(a);

  u64 start;
  u64 stop;
  u64 diff;
  
  // preparing file
  ofstream outputfile;
  char path[256];
  sprintf(path, "results/level2/%d/matrix_vector_mul_f48_nonSSE.txt", size);
  outputfile.open (path);

  for ( int i = 0; i < runs; i++ ) {
    start = rdtsc();
    matrix_vector_mul_f48(matrix,a);
    stop = rdtsc();
    diff = stop - start;
    outputfile<<diff<<endl;
    // repopulate 
    populate_matrix(matrix);
    populate_array(a);
  }
  outputfile.close();
  cout<<"Done. Results in results/level2/"<<size<<"/matrix_vector_mul_f48_nonSSE.txt"<<endl;
}

void test_matrix_vector_mul_double_v1_SSE()
{
  cout<<"DOUBLE(SSE-v1) matrix-vector multiplication..." << endl;
  double* a = new double[size];
  double** matrix = new double*[size];
  populate_matrix(matrix);
  populate_array(a);

  u64 start;
  double dot_prod;
  u64 stop;
  u64 diff;
  
  // preparing file
  ofstream outputfile;
  char path[256];
  sprintf(path, "results/level2/%d/matrix_vector_mul_double_v1.txt", size);
  outputfile.open (path);

  for ( int i = 0; i < runs; i++ ) {
    start = rdtsc();
    matrix_vector_mul_SSE_double(matrix,a);
    stop = rdtsc();
    diff = stop - start;
    outputfile<<diff<<endl;
    // repopulate 
    populate_matrix(matrix);
    populate_array(a);
  }
  outputfile.close();
  cout<<"Done. Results in results/level2/"<<size<<"/matrix_vector_mul_double_v1.txt"<<endl;
}

void test_matrix_vector_mul_f48_v1_SSE() 
{
  cout<<"F48(SSE-v1) matrix-vector multiplication..." << endl;
  f48* a = new f48[size];
  f48** matrix = new f48*[size];
  populate_matrix(matrix);
  populate_array(a);

  u64 start;
  u64 stop;
  u64 diff;
  
  // preparing file
  ofstream outputfile;
  char path[256];
  sprintf(path, "results/level2/%d/matrix_vector_mul_f48_v1.txt", size);
  outputfile.open (path);

  for ( int i = 0; i < runs; i++ ) {
    start = rdtsc();
    matrix_vector_mul_SSE_f48(matrix,a);
    stop = rdtsc();
    diff = stop - start;
    outputfile<<diff<<endl;
    // repopulate 
    populate_matrix(matrix);
    populate_array(a);
  }
  outputfile.close();
  cout<<"Done. Results in results/level2/"<<size<<"/matrix_vector_mul_f48_v1.txt"<<endl;
}

void test_matrix_vector_mul_double_v2_SSE()
{
  cout<<"DOUBLE(SSE-v2) matrix-vector multiplication..." << endl;
  double* a = new double[size];
  double** matrix = new double*[size];
  populate_matrix(matrix);
  populate_array(a);

  u64 start;
  double dot_prod;
  u64 stop;
  u64 diff;
  
  // preparing file
  ofstream outputfile;
  char path[256];
  sprintf(path, "results/level2/%d/matrix_vector_mul_double_v2.txt", size);
  outputfile.open (path);

  for ( int i = 0; i < runs; i++ ) {
    start = rdtsc();
    matrix_vector_mul_SSE_double_v2(matrix,a);
    stop = rdtsc();
    diff = stop - start;
    outputfile<<diff<<endl;
    // repopulate 
    populate_matrix(matrix);
    populate_array(a);
  }
  outputfile.close();
  cout<<"Done. Results in results/level2/"<<size<<"/matrix_vector_mul_double_v2.txt"<<endl;
}

void test_matrix_vector_mul_f48_v2_SSE()
{
  cout<<"F48(SSE-v2) matrix-vector multiplication..." << endl;
  f48* a = new f48[size];
  f48** matrix = new f48*[size];
  populate_matrix(matrix);
  populate_array(a);

  u64 start;
  u64 stop;
  u64 diff;
  
  // preparing file
  ofstream outputfile;
    char path[256];
  sprintf(path, "results/level2/%d/matrix_vector_mul_f48_v2.txt", size);
  outputfile.open (path);

  for ( int i = 0; i < runs; i++ ) {
    start = rdtsc();
    matrix_vector_mul_SSE_f48_v2(matrix,a);
    stop = rdtsc();
    diff = stop - start;
    outputfile<<diff<<endl;
    // repopulate 
    populate_matrix(matrix);
    populate_array(a);
  }
  outputfile.close();
  cout<<"Done. Results in results/level2/"<<size<<"/matrix_vector_mul_f48_v2.txt"<<endl;
}

void test_matrix_vector_mul_f48_loopunroll_SSE()
{
  cout<<"F48(SSE-loop-unroll) matrix-vector multiplication..." << endl;
  f48* a = new f48[size];
  f48** matrix = new f48*[size];
  populate_matrix(matrix);
  populate_array(a);

  u64 start;
  u64 stop;
  u64 diff;
  
  // preparing file
  ofstream outputfile;
    char path[256];
  sprintf(path, "results/level2/%d/matrix_vector_mul_f48_loopunroll.txt", size);
  outputfile.open (path);

  for ( int i = 0; i < runs; i++ ) {
    start = rdtsc();
    matrix_vector_mul_SSE_f48_loop_unrolled(matrix,a);
    stop = rdtsc();
    diff = stop - start;
    outputfile<<diff<<endl;
    // repopulate 
    populate_matrix(matrix);
    populate_array(a);
  }
  outputfile.close();
  cout<<"Done. Results in results/level2/"<<size<<"/matrix_vector_mul_f48_loopunroll.txt"<<endl;
}
    /*** END LEVEL ***/

    /*** LEVEL 3 ***/
// TODO: REQUIRES IMPLEMENTATION OF MAT-MAT MUL! 


/************ END TESTS ************/

/************ REPORTS ************/

    /*** LEVEL 1 REPORTS ***/
void build_report_blas1()
{
  string double_dot_prod[runs];
  string double_scale[runs];
  string double_max[runs];
  string double_min[runs];
  string double_magnitude[runs];
  string f48_dot_prod[runs];
  string f48_scale[runs];
  string f48_max[runs];
  string f48_min[runs];
  string f48_magnitude[runs];

  string line;
  char path[256];
  sprintf(path, "results/level1/%d/dot_prod_SSE_double.txt", size);
  ifstream test_double_dot_prod (path);
  int i=0;// counter
  if (test_double_dot_prod.is_open()) {
    while(getline(test_double_dot_prod, line)){
      double_dot_prod[i] = line;
      i++;
    }
    test_double_dot_prod.close();
  } else cout<<"ERROR: Unable to open file (results/level1/"<<size<<"/dot_prod_SSE_double.txt)";
  char path2[256];
  sprintf(path2, "results/level1/%d/scale_SSE_double.txt", size);
  ifstream test_double_scale (path2);
  i=0;// counter
  if (test_double_scale.is_open()) {
    while(getline(test_double_scale, line)){
      double_scale[i] = line;
      i++;
    }
    test_double_scale.close();
  } else cout<<"ERROR: Unable to open file (results/level1/"<<size<<"/scale_SSE_double.txt)";
  char path3[256];
  sprintf(path3, "results/level1/%d/max_SSE_double.txt", size);
  ifstream test_double_max (path3);
  i=0;// counter
  if (test_double_max.is_open()) {
    while(getline(test_double_max, line)){
      double_max[i] = line;
      i++;
    }
    test_double_max.close();
  } else cout<<"ERROR: Unable to open file (results/level1/"<<size<<"/max_SSE_double.txt)";
  char path4[256];
  sprintf(path4, "results/level1/%d/min_SSE_double.txt", size);
  ifstream test_double_min (path4);
  i=0;// counter
  if (test_double_min.is_open()) {
    while(getline(test_double_min, line)){
      double_min[i] = line;
      i++;
    }
    test_double_min.close();
  } else cout<<"ERROR: Unable to open file (results/level1/"<<size<<"/min_SSE_double.txt)";
  char path5[256];
  sprintf(path5, "results/level1/%d/magnitude_SSE_double.txt", size);
  ifstream test_double_magnitude (path5);
  i=0;// counter
  if (test_double_magnitude.is_open()) {
    while(getline(test_double_magnitude, line)){
      double_magnitude[i] = line;
      i++;
    }
    test_double_magnitude.close();
  } else cout<<"ERROR: Unable to open file (results/level1/"<<size<<"/magnitude_SSE_double.txt)";
  char path6[256];
  sprintf(path6, "results/level1/%d/dot_prod_SSE_f48.txt", size);
  ifstream test_f48_dot_prod (path6);
  i=0;// counter
  if (test_f48_dot_prod.is_open()) {
    while(getline(test_f48_dot_prod, line)){
      f48_dot_prod[i] = line;
      i++;
    }
    test_f48_dot_prod.close();
  } else cout<<"ERROR: Unable to open file (results/level1/"<<size<<"/dot_prod_SSE_f48.txt)";
  char path7[256];
  sprintf(path7, "results/level1/%d/scale_SSE_f48.txt", size);
  ifstream test_f48_scale (path7);
  i=0;// counter
  if (test_f48_scale.is_open()) {
    while(getline(test_f48_scale, line)){
      f48_scale[i] = line;
      i++;
    }
    test_f48_scale.close();
  } else cout<<"ERROR: Unable to open file (results/level1/"<<size<<"/scale_SSE_f48.txt)";
  char path8[256];
  sprintf(path8, "results/level1/%d/max_SSE_f48.txt", size);
  ifstream test_f48_max (path8);
  i=0;// counter
  if (test_f48_max.is_open()) {
    while(getline(test_f48_max, line)){
      f48_max[i] = line;
      i++;
    }
    test_f48_max.close();
  } else cout<<"ERROR: Unable to open file (results/level1/"<<size<<"/max_SSE_f48.txt)";
  char path9[256];
  sprintf(path9, "results/level1/%d/min_SSE_f48.txt", size);
  ifstream test_f48_min (path9);
  i=0;// counter
  if (test_f48_min.is_open()) {
    while(getline(test_f48_min, line)){
      f48_min[i] = line;
      i++;
    }
    test_f48_min.close();
  } else cout<<"ERROR: Unable to open file (results/level1/"<<size<<"/min_SSE_f48.txt)";
  char path10[256];
  sprintf(path10, "results/level1/%d/magnitude_SSE_f48.txt", size);
  ifstream test_f48_magnitude (path10);
  i=0;// counter
  if (test_f48_magnitude.is_open()) {
    while(getline(test_f48_magnitude, line)){
      f48_magnitude[i] = line;
      i++;
    }
    test_f48_magnitude.close();
  } else cout<<"ERROR: Unable to open file (results/level1/"<<size<<"/magnitude_SSE_f48.txt)";
// all results loaded into memory - require structure of final report
// report for dot prod with magnitude
  ofstream dot_prod_mag_report;
  sprintf(path, "results/level1/reports/%d/dot_product_and_magnitude.csv", size);
  dot_prod_mag_report.open (path);
  dot_prod_mag_report<<"Dot product,,Magnitude"<<endl;
  dot_prod_mag_report<<"Double,F48,Double,F48"<<endl;
  for(i=0; i<runs; i++){
    dot_prod_mag_report<<double_dot_prod[i]<<","<<f48_dot_prod[i]<<","<<double_magnitude[i]<<","<<f48_magnitude[i]<<endl;
  }
  dot_prod_mag_report.close();
  cout<<"Dot-product and magnitude report built sucessfully. (results/level1/reports/"<<size<<"/dot_product_and_magnitude.csv)"<<endl;
// report for max with min
  ofstream max_min_report;
  sprintf(path2, "results/level1/reports/%d/maximum_and_minimum.csv", size);
  max_min_report.open (path2);
  max_min_report<<"Absolute Maximum,,Absolute Minimum"<<endl;
  max_min_report<<"Double,F48,Double,F48"<<endl;
  for(i=0; i<runs; i++){
    max_min_report<<double_max[i]<<","<<f48_max[i]<<","<<double_min[i]<<","<<f48_min[i]<<endl;
  }
  max_min_report.close();
  cout<<"Absolute maximum and absolute minimum report built sucessfully. (results/level1/reports/"<<size<<"/maximum_and_minimum.csv)"<<endl;
// report for scale - one of the implemented BLAS L1 functions that makes changes to the vector itself
  ofstream scale_report;
  sprintf(path3, "results/level1/reports/%d/scale.csv", size);
  scale_report.open (path3);
  scale_report<<"Scale"<<endl;
  scale_report<<"Double,F48,F48-Double"<<endl;
  for(i=0; i<runs; i++){
    scale_report<<double_scale[i]<<","<<f48_scale[i]<<endl;
  }
  scale_report.close();
  cout<<"Scale report built sucessfully. (results/level1/reports/"<<size<<"/scale.csv)"<<endl;

  cout<<"BLAS level 1 reports completed. (results/level1/reports/"<<size<<"/)"<<endl;
}

    /*** LEVEL 2 REPORTS ***/
// FUNCTION REQUIRES TEST RESULTS FROM DOUBLE AND F48 IN THE PROPER FILES
void build_report_matrix_vector_v1_v2_nonSSE_double_vs_f48()
{
  string double_results_nonSSE[runs];
  string double_results_SSEv1[runs];
  string double_results_SSEv2[runs];
  string f48_results_nonSSE[runs];
  string f48_results_SSEv1[runs];
  string f48_results_SSEv2[runs];
  string f48_results_SSE_loopunroll[runs];
  
  // populating results from files
  string line;
  
  // double results (nonsse, ssev1, ssev2)
  char path[256];
  sprintf(path, "results/level2/%d/matrix_vector_mul_double_nonSSE.txt", size);
  ifstream test_nonSSE (path);
  int i=0;
  if (test_nonSSE.is_open())
  {
    while ( getline (test_nonSSE,line) )
    {
      double_results_nonSSE[i] = line;
      i++;
    }
    test_nonSSE.close();
  } else cout << "ERROR: Unable to open file (results/level2/"<<size<<"/matrix_vector_mul_double_nonSSE.txt)";
  char path2[256];
  sprintf(path2, "results/level2/%d/matrix_vector_mul_double_v1.txt", size);
  ifstream test_SSEv1 (path2);
  i=0; // reset count
  if (test_SSEv1.is_open())
  {
    while ( getline (test_SSEv1,line) )
    {
      double_results_SSEv1[i] = line;
      i++;
    }
    test_SSEv1.close();
  } else cout << "ERROR: Unable to open file (results/level2/"<<size<<"/matrix_vector_mul_double_v1.txt)"; 
  sprintf(path2, "results/level2/%d/matrix_vector_mul_double_v2.txt", size);
  ifstream test_SSEv2 (path2);
  i=0; // reset count
  if (test_SSEv2.is_open())
  {
    while ( getline (test_SSEv2,line) )
    {
      double_results_SSEv2[i] = line;
      i++;
    }
    test_SSEv2.close();
  } else cout << "ERROR: Unable to open file (results/level2/"<<size<<"/matrix_vector_mul_double_v2.txt)"; 
    char path3[256];
  sprintf(path3, "results/level2/%d/matrix_vector_mul_f48_nonSSE.txt", size);
  // f48 results (nonsse, ssev1, ssev2, sseloopunroll)
  ifstream test_nonSSE_f48 (path3);
  i=0;
  if (test_nonSSE_f48.is_open())
  {
    while ( getline (test_nonSSE_f48,line) )
    {
      f48_results_nonSSE[i] = line;
      i++;
    }
    test_nonSSE_f48.close();
  } else cout << "ERROR: Unable to open file (results/level2/"<<size<<"/matrix_vector_mul_f48_nonSSE.txt)";
  char path4[256];
  sprintf(path4, "results/level2/%d/matrix_vector_mul_f48_v1.txt", size);
  ifstream test_SSEv1_f48 (path4);
  i=0; // reset count
  if (test_SSEv1_f48.is_open())
  {
    while ( getline (test_SSEv1_f48,line) )
    {
      f48_results_SSEv1[i] = line;
      i++;
    }
    test_SSEv1_f48.close();
  } else cout << "ERROR: Unable to open file (results/level2/"<<size<<"/matrix_vector_mul_f48_v1.txt)"; 
  char path5[256];
  sprintf(path5, "results/level2/%d/matrix_vector_mul_f48_v2.txt", size);
  ifstream test_SSEv2_f48 (path5);
  i=0; // reset count
  if (test_SSEv2_f48.is_open())
  {
    while ( getline (test_SSEv2_f48,line) )
    {
      f48_results_SSEv2[i] = line;
      i++;
    }
    test_SSEv2_f48.close();
  } else cout << "ERROR: Unable to open file (results/level2/"<<size<<"/matrix_vector_mul_f48_v2.txt)"; 
  char path6[256];
  sprintf(path6, "results/level2/%d/matrix_vector_mul_f48_loopunroll.txt", size);
  ifstream test_SSEloop_f48 (path6);
  i=0; // reset count
  if (test_SSEloop_f48.is_open())
  {
    while ( getline (test_SSEloop_f48,line) )
    {
      f48_results_SSE_loopunroll[i] = line;
      i++;
    }
    test_SSEloop_f48.close();
  } else cout << "ERROR: Unable to open file (results/level2/"<<size<<"/matrix_vector_mul_f48_loopunroll.txt)"; 
  
  
  
  ofstream myfile;
  sprintf(path, "results/level2/reports/%d/matrix_vector_mul_report.csv", size);
  myfile.open (path);
  
  myfile<<"Matrix-vector Multiplication,,,,,,"<<endl;
  myfile<<"Double,,,F48,,,"<<endl;
  myfile<<"non-SSE,SSE-v1,SSE-v2,non-SSE,SSE-v1,SSE-v2,SSE-loop un-rolled"<<endl;
  // TODO: refactor to new structure ^^
  for(i=0;i<runs;i++){
    myfile<<double_results_nonSSE[i]<<","<<double_results_SSEv1[i]<<","<<double_results_SSEv2[i]<<","<<f48_results_nonSSE[i]<<","<<f48_results_SSEv1[i]<<","<<f48_results_SSEv2[i]<<","<<f48_results_SSE_loopunroll[i]<<endl;
  }
  myfile.close();
  cout<<"Matrix-vector multiplication results compiled for non-SSE,SSEv1,SSEv2. (results/level2/reports/"<<size<<"/matrix_vector_mul_report.txt)"<<endl;
}

    /*** LEVEL 3 REPORTS ***/
  // TODO NEEDS ADDITION AFTER IMPLEMENTATION AND TESTS

/************ END REPORTS ************/


/************* MAIN ***********************/

int main(int argc, char* argv[])
{



//  u48 dummy48u;
//  u64 dummy64u;

//  double dummy64d;
//
//  test_type(dummy48u, "u48     ");
//  test_type(dummy64u, "u64     ");
//  test_u48_vec();
//  test_type(dummy64d, "f64     ");
// test_type(dummy48f, "f48     ");
//  test_f48_vec();
//  test_double_vec();
//double a;
//cin>>a;
//f48 dummy48f (a);
//cin>>a;

  
// int x;
// f48 a[8];
//  a[0] = f48(1.5);
//  a[1] = f48(64.9);
//  a[2] = f48(12.8);
//  a[3] = f48(1.4);
//  a[4] = f48(3.2);
//  a[5] = f48(2.1);
//  a[6] = f48(1.89);
//  a[7] = f48(1.3);
//  a[2] = f48(2.1);
// a[3] = f48(7.1);
//  double b[2];
//  b[0] = 4.25649874;
//  b[1] = 122.216548;
//   double c[2];
//  c[0] = 3.8;
//  c[1] = 3.1;
// //b[3] = f48(1.5);
// double result;
// result = dot_product_SSE_double(a,b);
// // need conversion from f48 to double
// cout<<"\n RESULT: "<<result;
// cin>>x;

// TESTING F48 dot prod and DOUBLE dot prod
//   test_f48_dot_prod();
//   test_double_dot_prod();
// TESTING F48 and DOUBLE SCALE
//   test_f48_scale();
//  test_double_scale();
 
//  test_max_f48_SSE();
//  test_max_double_SSE();
//  test_min_f48_SSE();
//  test_min_double_SSE();

  
//   magnitude_SSE_double(b);
 
//  test_magnitude_f48_SSE();
//  test_magnitude_double_SSE();
//  build_report_magnitude();

 
 
//  test_f48_scale();
//  test_double_scale();
//  build_report_scaling();
 
// TEST TEST TEST
// f48** matrixa = new f48*[8];
// for(int i = 0; i < 8; ++i){
//     matrixa[i] = new f48[8];
// }

// for(unsigned i=0;i<8;i++) {
//     for(unsigned j=0;j<8;j++) {
//         matrixa[i][j] = f48(j);
//     }
// }

// f48** matrixb = new f48*[8];
// for(int i = 0; i < 8; ++i){
//     matrixb[i] = new f48[8];
// }

// for(unsigned i=0;i<8;i++) {
//     for(unsigned j=0;j<8;j++) {
//         matrixb[i][j] = f48(j);
//     }
// }

// cout<<endl<<"MATRIX A"<<endl; 
// for(unsigned i=0;i<8;i++) {
//     for(unsigned j=0;j<8;j++) {
//         cout<<double(matrixa[i][j])<<"\t";
//     }
//     cout<<endl;
// }

// cout<<endl<<"MATRIX B"<<endl;

// for(unsigned i=0;i<8;i++) {
//     for(unsigned j=0;j<8;j++) {
//         cout<<double(matrixb[i][j])<<"\t";
//     }
//     cout<<endl;
// }

// matrixa = matrix_matrix_mul_f48(matrixa,matrixb); 

// cout<<endl<<"RESULT!!!!"<<endl;
// for(unsigned i=0;i<8;i++) {
//     for(unsigned j=0;j<8;j++) {
//         cout<<double(matrixa[i][j])<<"\t";
//     }
//     cout<<endl;
// }



// 
// f48* vector = new f48[8];
// for(unsigned i=0;i<8;i++){
//   vector[i] = f48(i);
// }
// for(unsigned i=0;i<8;i++){
//   cout <<(double)vector[i]<<endl;
// }
// 
// u64 start;
// u64 stop;
// u64 v1;
// u64 v2;
// u64 v2v1;
// // matrix_vector_mul_SSE_f48(matrix,vector);
// 
// start = rdtsc();
// matrix_vector_mul_SSE_f48_loop_unrolled(matrix,vector);
// stop = rdtsc();
// v1 = stop-start;
// cout<<endl<<endl;
// for(unsigned i=0;i<8;i++){
//   cout <<(double)vector[i]<<endl;
// }



 //TESTING
// test_matrix_vector_mul_double_v1_SSE();
// test_matrix_vector_mul_double_v2_SSE();
// test_matrix_vector_mul_double_nonSSE();
//  build_report_matrix_vector_v1_v2_nonSSE_double();

//  test_matrix_vector_mul_f48_v1_SSE();
// test_matrix_vector_mul_f48_v2_SSE();
 
 // TESTING
//  test_matrix_vector_mul_f48_nonSSE();
//  test_matrix_vector_mul_f48_loopunroll_SSE();
 
// start = rdtsc();
// matrix_vector_mul_SSE_double_v2(matrix,vector);
// stop = rdtsc();
// v2 = stop-start;

// for(unsigned i=0;i<4;i++){
//   cout << vector[i]<<endl;
// }
/*
cout<<"V1: "<<v1<<endl;
cout<<"V2: "<<v2<<endl;
cout<<"V2-V1: "<<v2-v1<<endl;*/

// int test;
// cin>>test;
cout<<"Creating folder structure required"<<endl;
char mkcmd[256];
char dirpath[80] = "./results/level1/reports";
sprintf(mkcmd, "mkdir -p %s", dirpath);
int res = system(mkcmd);
if(res<0){
  cout<<"ERROR occured in creating folder structure.";
  return -1;
}
char dirpathtwo[80] = "./results/level2/reports";
sprintf(mkcmd, "mkdir -p %s", dirpathtwo);
res = system(mkcmd);
if(res<0){
  cout<<"ERROR occured in creating folder structure.";
  return -1;
}
char dirpaththree[80] = "./results/level3/reports";
sprintf(mkcmd, "mkdir -p %s", dirpaththree);
res = system(mkcmd);
if(res<0){
  cout<<"ERROR occured in creating folder structure.";
  return -1;
}

// TESTING MENU!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
int option=0;
int inneroption=0;
do { 
  cout << "Size selected is: "<<size<<endl;
  cout << "Number of runs for each test is: "<<runs<<endl;
  cout << "1) BLAS level 1" << endl;
  cout << "2) BLAS level 2" << endl;
  cout << "3) BLAS level 3" << endl;
  cout << "4) Change size" << endl;
  cout << "5) Change number of runs" << endl;
  cout << "6) Exit " << endl;
  //Prompting user to enter an option according to menu
  cout << "Please select an option: ";
  cin >> option;  // taking option value as input and saving in variable "option"
  char *path = (char *)"";
  
  switch(option) {
    case 1: cout<<"BLAS LEVEL 1 SELECTED!"<<endl;
      do{
        cout << "1) Double tests" << endl;
        cout << "2) F48 tests" << endl;
        cout << "3) Build comparison reports" << endl;
        cout << "4) Back " << endl;
        cout << "Please select an option: ";
        cin >> inneroption;  // taking option value as input and saving in variable "option"
        switch(inneroption) {
          case 1: 
            cout<<"Creating folder structure"<<endl;
            path = (char *)"./results/level1/";
            sprintf(mkcmd, "mkdir -p %s/%d", path,size);
            res = system(mkcmd);
            if(res<0){
              cout<<"ERROR occured in creating folder structure.";
              return -1;
            }
            path = (char *)"./results/level1/reports/";
            sprintf(mkcmd, "mkdir -p %s/%d", path,size);
            res = system(mkcmd);
            if(res<0){
              cout<<"ERROR occured in creating folder structure.";
              return -1;
            }
            cout<<"Running dot-prod, scale, max/min, magnitude tests on doubles"<<endl;
            test_double_dot_prod();
            cout<<endl;
            test_double_scale();
            cout<<endl;
            test_max_double_SSE();
            cout<<endl;
            test_min_double_SSE();
            cout<<endl;
            test_magnitude_double_SSE();
            cout<<endl;
            break;
          case 2:
            cout<<"Creating folder structure"<<endl;
            path = (char *)"./results/level1/";
            sprintf(mkcmd, "mkdir -p %s/%d", path,size);
            res = system(mkcmd);
            if(res<0){
              cout<<"ERROR occured in creating folder structure.";
              return -1;
            }
            path = (char *)"./results/level1/reports/";
            sprintf(mkcmd, "mkdir -p %s/%d", path,size);
            res = system(mkcmd);
            if(res<0){
              cout<<"ERROR occured in creating folder structure.";
              return -1;
            }
            cout<<"Running dot-prod, scale, max/min, magnitude tests on f48"<<endl;
            test_f48_dot_prod();
            cout<<endl;
            test_f48_scale();
            cout<<endl;
            test_max_f48_SSE();
            cout<<endl;
            test_min_f48_SSE();
            cout<<endl;
            test_magnitude_f48_SSE();
            cout<<endl;
            break;
          case 3: 
            cout<<"Creating folder structure"<<endl;
            path = (char *)"./results/level1/";
            sprintf(mkcmd, "mkdir -p %s/%d", path,size);
            res = system(mkcmd);
            if(res<0){
              cout<<"ERROR occured in creating folder structure.";
              return -1;
            }
            path = (char *)"./results/level1/reports/";
            sprintf(mkcmd, "mkdir -p %s/%d", path,size);
            res = system(mkcmd);
            if(res<0){
              cout<<"ERROR occured in creating folder structure.";
              return -1;
            }
            cout<<"Compiling results in the BLAS level 1 report... "<<endl;
            build_report_blas1(); // TODO - implement report gen
            cout<<endl;
            break;
          case 4: cout<<"Going back to main menu"<<endl<<endl;
            break;
          default: cout<<"try again";
            break;
        }
      }while(inneroption != 4);
      break;
    case 2: 
      cout<<endl<<"BLAS LEVEL 2 SELECTED"<<endl;
      do{
      	cout << "1) Double tests" << endl;
      	cout << "2) F48 tests" << endl;
      	cout << "3) Build comparison report" << endl;
      	cout << "4) Back " << endl;
      	cout << "Please select an option: ";
      	cin >> inneroption;  // taking option value as input and saving in variable "option"
      	 switch(inneroption) {
      	   case 1:
            cout<<"Creating folder structure"<<endl;
            path = (char *)"./results/level2/";
            sprintf(mkcmd, "mkdir -p %s/%d", path,size);
            res = system(mkcmd);
            if(res<0){
              cout<<"ERROR occured in creating folder structure.";
              return -1;
            }
            path = (char *)"./results/level2/reports/";
            sprintf(mkcmd, "mkdir -p %s/%d", path,size);
            res = system(mkcmd);
            if(res<0){
              cout<<"ERROR occured in creating folder structure.";
              return -1;
            }
      	     cout<<endl<<"Running matrix-vector tests on doubles"<<endl;
              test_matrix_vector_mul_double_nonSSE();       	     
      	     cout<<endl;
      	     test_matrix_vector_mul_double_v1_SSE();
      	     cout<<endl;
      	     test_matrix_vector_mul_double_v2_SSE();
      	     cout<<endl;
      	     break;
      	   case 2:
              cout<<"Creating folder structure"<<endl;
              path = (char *)"./results/level2/";
              sprintf(mkcmd, "mkdir -p %s/%d", path,size);
              res = system(mkcmd);
              if(res<0){
                cout<<"ERROR occured in creating folder structure.";
                return -1;
              }
              path = (char *)"./results/level2/reports/";
              sprintf(mkcmd, "mkdir -p %s/%d", path,size);
              res = system(mkcmd);
              if(res<0){
                cout<<"ERROR occured in creating folder structure.";
                return -1;
              }  
      	     cout<<endl<<"Running matrix-vector tests on f48"<<endl;
             test_matrix_vector_mul_f48_nonSSE(); 
      	     cout<<endl;
      	     test_matrix_vector_mul_f48_v1_SSE();
      	     cout<<endl;
      	     test_matrix_vector_mul_f48_v2_SSE();
      	     cout<<endl;
      	     test_matrix_vector_mul_f48_loopunroll_SSE();
      	     cout<<endl;
      	     break;
      	   case 3:
              cout<<"Creating folder structure"<<endl;
              path = (char *)"./results/level2/";
              sprintf(mkcmd, "mkdir -p %s/%d", path,size);
              res = system(mkcmd);
              if(res<0){
                cout<<"ERROR occured in creating folder structure.";
                return -1;
              }
              path = (char *)"./results/level2/reports/";
              sprintf(mkcmd, "mkdir -p %s/%d", path,size);
              res = system(mkcmd);
              if(res<0){
                cout<<"ERROR occured in creating folder structure.";
                return -1;
              }
      	     cout<<endl<<"Compiling results in matrix-vector multiplication report..."<<endl;
      	     build_report_matrix_vector_v1_v2_nonSSE_double_vs_f48();
      	     cout<<endl;
      	     break;
      	   case 4: cout<<"Going back to main menu"<<endl<<endl;
      	     break;
      	   default: cout<<"try again";
      	     break;
      	 }
      }while(inneroption != 4);
      break;
    case 3: 
      cout<<"BLAS LEVEL 3 SELECTED!"<<endl;
      break;
    case 4: 
      cout<<"Enter the new value of the size (current size: "<<size<<", enter 0 (or<0) to keep the current value"<<endl;
      int temp_size;
      cin >> temp_size;
      if(temp_size>0) {
        size = temp_size;
        cout<<"Size sucessfully changed to: "<<size<<endl;
      } else {
        cout<<"New size negative or 0 not accepted - current size remains the same ("<<size<<")"<<endl;
      }
      break;
    case 5:
      cout<<"Enter the number of runs (current number of runs: "<<runs<<", enter 0 (or<0) to keep the current value"<<endl;
      int temp_runs;
      cin>>temp_runs;
      if(temp_runs>0){
        runs = temp_runs;
        cout<<"Number of runs changed to: "<<runs<<endl;
      } else {
        cout<<"New number of runs negative or 0 not accepted - current number of runs remains the same ("<<runs<<")"<<endl;
      }
      break;
    case 6: cout<<"Exiting program"<<endl;
      return 0;
      break;
    default: cout<<"Unknown option, please try again!"<<endl;
      break;
  }

} while(option != 6);



 
 return 0;
}