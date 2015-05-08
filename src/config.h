#ifndef CONFIG_H
#define CONFIG_H
struct stat st = {0};

int size = 256;
char sizestr[256];
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


#else
#endif
