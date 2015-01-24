#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib> 
#include <string>
#include <x86intrin.h>
#include <stdio.h>
#include <bitset>


using namespace std;

const int size = 1024 * 1024;

typedef unsigned long long u64;
typedef unsigned short u16;


template<typename T>
void show_binrep(const T& a)
{
    const char* beg = reinterpret_cast<const char*>(&a);
    const char* end = beg + sizeof(a);
	std::cout<<" => ";
    while(beg != end)
        std::cout << std::bitset<CHAR_BIT>(*beg++) << ' ';
    std::cout << '\n';
}



union conversion_union {
  long long l;
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

union un {
  double f;
  u64 u;
};

#define INTEL_TYPE_CONVERSIONS 1
#if INTEL_TYPE_CONVERSIONS
f48::f48(double value)
{
  // convert to 64-bit pattern
//   u64 tmp = _castf64_u64(value);// not valid INTEL operation
  convert.d = value;
  long long tmp = 0;
//  convert.l = 5558888888888888888;
	cout<<value;
	cout<<"\n";
  cout<<convert.d;
  show_binrep(convert.d);
  // round to nearest even is a little complex
  // the u64 number has the following format:
  // 47upper_bits:L:G:15lower_bits
  tmp = convert.l;
  bool S = (tmp & 65535)>0 ? 1:0; // STICKY bit - OR of the remaining bits
  bool R = (tmp >> 15) & 1;  // ROUND bit - first bit removed
  bool G = (tmp >> 16) & 1;  // GUARD bit - LSB of result
  
  u64 mantissa = (tmp&4503599627370495)>>16; // extracts the remainding mantissa from number
  long long signExponent = (tmp>>52)<<52; // clear the mantissa
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
cout<<G;
show_binrep(G);
  if(G){
	  cout<<(R|S);
	  cout<<"::::";
	  cout<<R||S;
	  if (R||S){ // R or S is true go up
		cout<<"DOING THIS MANTISA MANIPULATION AS R OR S IS 1";
		  mantissa = mantissa +1;
		  if((mantissa>>36)&1){ // overflow of mantissa
			  mantissa = 0; // reset mantissa
			  // add one to exponent
			  // TODO: extract exponent only add one to it and check for overflow
			  // if overflow happens on exponent set number to positive/negative infinity
			  signExponent = signExponent + 1;
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
				  // TODO: extract exponent only add one to it and check for overflow
				  // if overflow happens on exponent set number to positive/negative infinity
				  signExponent = signExponent + 1;
			  }
		  } // else do nothing 
	  }
  } else {
	  // not required as if G is 0 R&S are x (don't cares)
  }
  long long result = (mantissa<<16) + signExponent;
  show_binrep(mantissa<<16);
  show_binrep(signExponent);
  show_binrep(result);
  
/* ------ Testing purposes ------ */  
  cout<<"S="<<S;
  cout<<"R="<<R;
  cout<<"G="<<G;
/* ------------------------------ */

// to convert the number back to double there is a need of having the conversion 
// done in the union (convert)

convert.l = result;
cout<<convert.d;
	// compensate for little endianess
  this->num = result >> 16;
  
}

// TODO: implementation of BLAS

f48::operator double()
{
  // convert to double
  convert.l = (this->num)<<16;
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
    cout << "f64 vec "<< diff << " " << sum << " "<< sizeof(u48) << endl;
  }
}



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
double a;
cin>>a;
f48 dummy48f (a);
cin>>a;
  return 0;
}
