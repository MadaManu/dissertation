#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib> 
#include <string>
#include <x86intrin.h>
#include <stdio.h>
#include <bitset>
#include <time.h>
#include <fstream>
#include <stdio.h>
#include <string.h>

using namespace std;

const int size = 1024 * 1024;

typedef unsigned long long u64;
typedef unsigned short u16;


//template<typename T>
//void show_binrep(const T& a)
//{
//    const char* beg = reinterpret_cast<const char*>(&a);
//    const char* end = beg + sizeof(a);
//	std::cout<<" => ";
//    while(beg != end)
//        std::cout << std::bitset<CHAR_BIT>(*beg++) << ' ';
//    std::cout << '\n';
//}



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

#define INTEL_TYPE_CONVERSIONS 1
#if INTEL_TYPE_CONVERSIONS
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

#else
f48::f48(double value)
{
  union un temp;
  u64 round;
	
  // convert to 64-bit pattern
  temp.f = value;

  // simple rounding
  temp.u = temp.u + ((1 << 15)-1);
  
  /*
  // round to nearest even
  round = temp.u;
  round = (round + ((1 << 15)-1)) | round << 1;
  need to look at that PhD thesis that describes rounding
  */

  // compensate for little-endianness
  this->num = temp.u >> 16;
}

f48::operator double()
{
  union un temp;
  double result;

  temp.u = this->num;

  // compensate for little-endianness
  temp.u = temp.u << 16;

  return temp.f;
}
#endif

/************************************************************/

template <class T>
void populate_array(T * a)
{
  //srand((unsigned)time(0)); 
  srand(5);

  for ( int i = 0; i < size; i++ ) {
    a[i] = rand() % 1024;
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

template <class T>
T sum_arrays(T * a, T * b, T * c)
{
  T result;
  for ( int i = 0; i < size; i++ ) {
    T sum = a[i] + b[i];
    c[i] = sum;
    result = result + sum;
  }
  return result;
}

u48 sum_arrays_SSE(u48 * a, u48 * b, u48 * c)
{
  u48 result;
  __m128i result_vec = _mm_set1_epi32(0);
  // insert zeroes in the two upper bytes
  const __m128i in_mask = _mm_set_epi8(255, 255, 11, 10, 9, 8,  7,  6,
				    255, 255, 5, 4, 3, 2, 1, 0);
  // I'm not quite sure that this out mask is correct
  const __m128i out_mask = _mm_set_epi8(0, 1, 2,   3,  4,  5,
					8, 9, 10, 11, 12, 13,
					255, 255, 255, 255);

  long long tmp[2];

  for ( int i = 0; i < size; i+= 2 ) {
    __m128i a_vec = _mm_loadu_si128((__m128i*)(&a[i]));
    a_vec = _mm_shuffle_epi8(a_vec, in_mask);
    __m128i b_vec = _mm_loadu_si128((__m128i*)(&b[i]));
    b_vec = _mm_shuffle_epi8(b_vec, in_mask);
    __m128i sum_vec = _mm_add_epi64(a_vec, b_vec);
    // need to shuffle data before writing out
    //cout << hex << a[i] + b[i] << " | " <<  a[i+1] + b[i+1] << endl;
    //u64 temp[2];
    //_mm_storeu_si128((__m128i*)temp, sum_vec);
    //cout << hex << temp[0] << " | " <<  temp[1] << endl;
    __m128i out_vec = _mm_shuffle_epi8(out_mask, sum_vec);
    _mm_storeu_si128((__m128i*)&c[i], out_vec);
    result_vec = _mm_add_epi64(result_vec, sum_vec);
  }
  _mm_storeu_si128((__m128i*)tmp, result_vec);
  result = tmp[0] + tmp[1];

  return result;
}

f48 sum_arrays_SSE_f48(f48 * a, f48 * b, f48 * c)
{
  f48 result;
  __m128d result_vec = _mm_set1_pd(0.0);

  // the masking is different for f48 compared to u48.
  // with f48 we want to insert zeroes for the two lower bytes
  __m128i mask = _mm_set_epi8(11, 10, 9, 8,  7,  6, 255, 255,
  			      5, 4, 3, 2, 1, 0, 255, 255);

  double tmp[2];

  for ( int i = 0; i < size; i+= 2 ) {
    __m128i a_vec = _mm_loadu_si128((__m128i*)(&a[i]));
    a_vec = _mm_shuffle_epi8(a_vec, mask);
    __m128i b_vec = _mm_loadu_si128((__m128i*)(&b[i]));
    b_vec = _mm_shuffle_epi8(b_vec, mask);
    __m128d sum_vec = _mm_add_pd((__m128d)a_vec, (__m128d)b_vec);
    // BUG: need to round data before writing out
    // BUG: need to shuffle data before writing out
    _mm_storeu_pd((double*)&c[i], sum_vec);
    result_vec = _mm_add_pd(result_vec, sum_vec);
  }
  _mm_storeu_pd(tmp, result_vec);
  result = tmp[0] + tmp[1];

  return result;
}

double sum_arrays_SSE_double(double * a, double * b, double *c)
{
  double result;
  __m128d result_vec = _mm_set1_pd(0.0);

  double tmp[2];
  for ( int i = 0; i < size; i+= 2 ) {
    __m128d a_vec = _mm_loadu_pd(&a[i]);
    __m128d b_vec = _mm_loadu_pd(&b[i]);
    __m128d sum_vec = _mm_add_pd(a_vec, b_vec);
    _mm_storeu_pd(&c[i], sum_vec);
    result_vec = _mm_add_pd(result_vec, sum_vec);
  }
  _mm_storeu_pd(tmp, result_vec);
  result = tmp[0] + tmp[1];

  return result;
}

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
    
    #define bofs(base, ofs) (((double*)(base))+ofs)

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


/***************** TEST FUNCTIONS **************/
template <class T>
void test_type(T dummy, string s)
{
  T * a = new T[size];
  T * b = new T[size];
  T * c = new T[size];

  populate_array(a);
  populate_array(b);

  u64 start;
  T sum;
  u64 stop;
  u64 diff;

  for ( int i = 0; i < 10; i++ ) {
    start = rdtsc();
    sum = sum_arrays(a, b, c);
    stop = rdtsc();
    diff = stop - start;
    cout << s << diff << " " << sum << " "<< sizeof(u48) << endl;
  }
}

void test_u48_vec()
{
  u48 * a = new u48[size];
  u48 * b = new u48[size];
  u48 * c = new u48[size];

  populate_array(a);
  populate_array(b);

  u64 start;
  u48 sum;
  u64 stop;
  u64 diff;

  for ( int i = 0; i < 10; i++ ) {
    start = rdtsc();
    sum = sum_arrays_SSE(a, b, c);
    stop = rdtsc();
    diff = stop - start;
    cout << "u48 vec "<< diff << " " << sum << " "<< sizeof(u48) << endl;
  }
}

void test_f48_vec()
{
  f48 * a = new f48[size];
  f48 * b = new f48[size];
  f48 * c = new f48[size];

  populate_array(a);
  populate_array(b);

  u64 start;
  f48 sum;
  u64 stop;
  u64 diff;

  for ( int i = 0; i < 10; i++ ) {
    start = rdtsc();
    sum = sum_arrays_SSE_f48(a, b, c);
    stop = rdtsc();
    diff = stop - start;
    cout << "f48 vec "<< diff << " " << sum << " "<< sizeof(u48) << endl;
  }
}

void test_double_vec()
{
  double * a = new double[size];
  double * b = new double[size];
  double * c = new double[size];

  populate_array(a);
  populate_array(b);

  u64 start;
  double sum;
  u64 stop;
  u64 diff;

  for ( int i = 0; i < 10; i++ ) {
    start = rdtsc();
    sum = sum_arrays_SSE_double(a, b, c);
    stop = rdtsc();
    diff = stop - start;
    cout << "f64 vec "<< diff << " " << sum << " "<< sizeof(double) << endl;
  }
}

// TODO: function to take size
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
	_mm_store1_pd(&total, result_vec);
	return total;
}

// TODO: function to take size
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
	_mm_store1_pd(&total, result_vec);
	f48 total_result (total);
	return total_result;
}

// TODO: function to take size
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
  _mm_store1_pd(&maximum, max_result);
  f48 max_f48 (maximum);
  return max_f48;
}

// TODO: function to take size
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
  _mm_store1_pd(&minimum, min_result);
  f48 max_f48 (minimum);
  return max_f48;
}

// TODO: function to take size
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
  _mm_store1_pd(&maximum, max_result);
  return maximum;
}

// TODO: function to take size
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
  _mm_store1_pd(&minimum, min_result);
  return minimum;
}

// TODO: function to take size
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
  _mm_store1_pd(&res, result_vec);
  return f48(res);
}

// TODO: function to take size
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
  _mm_store1_pd(&res, result_vec);
  return res;
}

void test_f48_dot_prod()
{
  cout<<"f48 dot product" << endl;
  cout<<"RDTSC diff" << endl;
  f48 * a = new f48[size];
  f48 * b = new f48[size];

  populate_array(a);
  populate_array(b);

  u64 start;
  f48 dot_prod;
  u64 stop;
  u64 diff;

  for ( int i = 0; i < 100; i++ ) {
    start = rdtsc();
    dot_prod = dot_product_SSE_f48(a, b);
    stop = rdtsc();
    diff = stop - start;
    cout << diff << ',';
  }
  cout<<endl;
}

void test_double_dot_prod()
{
  cout<<"DOUBLE dot product" << endl;
  cout<<"RDTSC diff" << endl;
  double * a = new double[size];
  double * b = new double[size];

  populate_array(a);
  populate_array(b);

  u64 start;
  double dot_prod;
  u64 stop;
  u64 diff;

  for ( int i = 0; i < 100; i++ ) {
    start = rdtsc();
    dot_prod = dot_product_SSE_double(a, b);
    stop = rdtsc();
    diff = stop - start;
    cout << diff << ',';
  }
}

void test_f48_scale()
{
  cout<<"f48 scaling SSE..." << endl;
  f48 * a = new f48[size];
  populate_array(a);
  srand(5);
  f48 scalar = f48(rand() % 1024);

  u64 start;
  u64 stop;
  u64 diff;
  ofstream myfile;
  myfile.open ("stoker_results/scale_SSE_f48.txt");

  for ( int i = 0; i < 100; i++ ) {
    start = rdtsc();
    scale_f48_vector_SSE(a, scalar);
    stop = rdtsc();
    diff = stop - start;
    myfile<<diff<<endl;
//     cout << diff << ",";
  }
  cout<<"Done. Results in stoker_results/scale_SSE_f48.txt "<<endl;
}

void test_double_scale()
{
  cout<<"double scaling SSE..." << endl;
  double * a = new double[size];
  srand(5);
  double scalar = rand() % 1024;

  populate_array(a);

  u64 start;
  u64 stop;
  u64 diff;
  ofstream myfile;
  myfile.open ("stoker_results/scale_SSE_double.txt");

  for ( int i = 0; i < 100; i++ ) {
    start = rdtsc();
    scale_double_vector_SSE(a, scalar);
    stop = rdtsc();
    diff = stop - start;
    myfile<<diff<<endl;
//     cout << diff << ",";
  }
  cout<<"Done. Results in stoker_results/scale_SSE_double.txt "<<endl;
//   cout<<endl;
}

void test_max_f48_SSE()
{
  cout<<"f48 max SSE" << endl;
  cout<<"RDTSC diff" << endl;
  f48 * a = new f48[size];

  populate_array(a);

  u64 start;
  f48 dot_prod;
  u64 stop;
  u64 diff;

  for ( int i = 0; i < 100; i++ ) {
    start = rdtsc();
    dot_prod = absolute_max_SSE_f48(a);
    stop = rdtsc();
    diff = stop - start;
    cout << diff << ',';
  }
  cout<<endl;
}

void test_min_f48_SSE()
{
  cout<<"f48 min SSE" << endl;
  cout<<"RDTSC diff" << endl;
  f48 * a = new f48[size];

  populate_array(a);

  u64 start;
  f48 dot_prod;
  u64 stop;
  u64 diff;

  for ( int i = 0; i < 100; i++ ) {
    start = rdtsc();
    dot_prod = absolute_min_SSE_f48(a);
    stop = rdtsc();
    diff = stop - start;
    cout << diff << ',';
  }
  cout<<endl;
}


void test_max_double_SSE()
{
  cout<<"DOUBLE max SSE" << endl;
  cout<<"RDTSC diff" << endl;
  double * a = new double[size];

  populate_array(a);

  u64 start;
  double dot_prod;
  u64 stop;
  u64 diff;

  for ( int i = 0; i < 100; i++ ) {
    start = rdtsc();
    dot_prod = absolute_max_SSE_double(a);
    stop = rdtsc();
    diff = stop - start;
    cout << diff << ',';
  }
}

void test_min_double_SSE()
{
  cout<<"DOUBLE min SSE" << endl;
  cout<<"RDTSC diff" << endl;
  double * a = new double[size];

  populate_array(a);

  u64 start;
  double dot_prod;
  u64 stop;
  u64 diff;

  for ( int i = 0; i < 100; i++ ) {
    start = rdtsc();
    dot_prod = absolute_min_SSE_double(a);
    stop = rdtsc();
    diff = stop - start;
    cout << diff << ',';
  }
}

void test_magnitude_f48_SSE()
{
  cout<<"f48 magnitude SSE..." << endl;
  f48 * a = new f48[size];

  populate_array(a);

  u64 start;
  f48 dot_prod;
  u64 stop;
  u64 diff;
  
  // preparing file
  ofstream myfile;
  myfile.open ("results/magnitude_SSE_f48.txt");
  
  for ( int i = 0; i < 100; i++ ) {
    start = rdtsc();
    dot_prod = magnitude_SSE_f48(a);
    stop = rdtsc();
    diff = stop - start;
    myfile<<diff<<endl;
//     cout << diff << ',';
  }
  myfile.close();
  cout<<"Done. Results in results/magnitude_SSE_f48.txt "<<endl;
}

void test_magnitude_double_SSE()
{
  cout<<"DOUBLE magnitude SSE..." << endl;
  double * a = new double[size];

  populate_array(a);

  u64 start;
  double dot_prod;
  u64 stop;
  u64 diff;
  
  // preparing file
  ofstream myfile;
  myfile.open ("results/magnitude_SSE_double.txt");

  for ( int i = 0; i < 100; i++ ) {
    start = rdtsc();
    dot_prod = magnitude_SSE_double(a);
    stop = rdtsc();
    diff = stop - start;
    myfile <<diff<<endl;
//     cout << diff << ',';
  }
  myfile.close();
  cout<<"Done. Results in results/magnitude_SSE_double.txt "<<endl;
}

void build_report_magnitude()
{
  cout<<"Compiling results in main magnitude report..."<<endl;
  ofstream myfile;
  myfile.open ("results/magnitude_report.txt");
  
  myfile<<"Magnitude test (f48 vs double)"<<endl;
  myfile<<"SSE f48, SSE double"<<endl;
  string results[100]; // TODO: use the test results number from global here!
  string line;
  ifstream mag_f48 ("results/magnitude_SSE_double.txt");
  int i=0;
  if (mag_f48.is_open())
  {
    while ( getline (mag_f48,line) )
    {
      results[i] = line;
      i++;
    }
    mag_f48.close();
  }
  
  ifstream mag_double ("results/magnitude_SSE_f48.txt");
  i=0;
  if (mag_double.is_open())
  {
    while ( getline (mag_double,line) )
    {
      myfile<<results[i]<<","<<line<<endl;
      i++;
    }
    mag_double.close();
  }

  else cout << "Unable to open file"; 

  
  myfile.close();
  cout<<"Magnitude results compiled for f48 and double. (results/magnitude_report.txt)"<<endl;
}


void build_report_scaling() 
{
  cout<<"Compiling results in main scaling report..."<<endl;
  ofstream myfile;
  myfile.open ("stoker_results/scale_report.txt");
  myfile<<"Scale test (f48 vs double)"<<endl;
  myfile<<"SSE f48, SSE double"<<endl;
  
  string results[100]; // TODO: use the test results number from global here!
  string line;
  ifstream scale_f48 ("stoker_results/scale_SSE_double.txt");
  int i=0;
  if (scale_f48.is_open())
  {
    while ( getline (scale_f48,line) )
    {
      results[i] = line;
      i++;
    }
    scale_f48.close();
  }
  
  ifstream scale_double ("stoker_results/scale_SSE_f48.txt");
  i=0;
  if (scale_double.is_open())
  {
    while ( getline (scale_double,line) )
    {
      myfile<<results[i]<<","<<line<<endl;
      i++;
    }
    scale_double.close();
  }

  else cout << "Unable to open file"; 

  
  myfile.close();
  cout<<"Scale results compiled for f48 and double. (stoker_results/scale_report.txt)"<<endl;
}

// TODO: add number of runs for test results global
/************* MAIN ***********************/
int main()
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
f48 a[8];
 a[0] = f48(1.5);
 a[1] = f48(64.9);
 a[2] = f48(12.8);
 a[3] = f48(1.4);
 a[4] = f48(3.2);
 a[5] = f48(2.1);
 a[6] = f48(1.89);
 a[7] = f48(1.3);
//  a[2] = f48(2.1);
// a[3] = f48(7.1);
 double b[2];
 b[0] = 4.25649874;
 b[1] = 122.216548;
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

 
 
 test_f48_scale();
 test_double_scale();
 build_report_scaling();
 
 return 0;
}
