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
    /*** LEVEL 3 ***/
// TODO ADD L3 implementation

      /** MATRIX MATRIX MUL double **/
double** matrix_matrix_mul_double(double** a, double** b)
{
  double** res = new double*[SIZE];
  for(int i=0; i<SIZE; i++) {
    res[i] = new double[SIZE];
  }
  for(int i=0; i<SIZE; i++) { // a row count
    for(int j=0; j<SIZE; j++) { // b column count
      double sum = 0; // init sum
      for(int k=0; k<SIZE; k++) { // k offset
        sum += a[i][k]*b[k][j];
      }
      res[i][j] = sum;
    }
  }
  return res;
}

      /** MATRIX MATRIX MUL fl48 **/
fl48** matrix_matrix_mul_f48(fl48** a, fl48** b) {
  fl48** res = new fl48*[SIZE];
  for(int i=0; i<SIZE; i++) {
    res[i] = new fl48[SIZE];
  }
  for(int i=0; i<SIZE; i++) {
    for(int j=0; j<SIZE; j++) {
      double sum = 0; // compute the sum in doubles
      for(int k=0; k<SIZE; k++) {
        sum += double(a[i][k])*double(b[k][j]);
      }
      // save back to memory in fl48 format
      res[i][j] = fl48(sum);
    }
  }
  return res;
}

// implement the naive method
// then implement a locality improved method

      /** MATRIX MATRIX MUL double SSE **/
double** matrix_matrix_mul_double_SSE(double** a, double** b) {
  double** res = new double*[4];
  for(int i=0; i<4; i++) {
    res[i] = new double[4];
  }
  for(int i=0; i<4; i++){
    __m128d sum[4]; // SIZE
    for(int x=0;x<4;x++){ // initialisation of final sum
      sum[x] = _mm_setzero_pd();
    }
    for(int j=0; j<4; j+=2){
      for(int offset=0; offset<4; offset+=2){
        __m128d elemA = _mm_load_pd(&a[i][j]);
        __m128d elemb1 = _mm_load_pd(&b[j][offset]);
        __m128d elemb2 = _mm_load_pd(&b[j+1][offset]);

        __m128i mask = _mm_set_epi8(255, 255, 255, 255, 255, 255, 255, 255,
                7 ,6 ,5, 4, 3, 2, 1, 0);
        __m128d temp = (__m128d)_mm_shuffle_epi8((__m128i)elemb1, mask);
        mask = _mm_set_epi8(7 ,6 ,5, 4, 3, 2, 1, 0,
            255, 255, 255, 255, 255, 255, 255, 255);
        __m128d temp2 = (__m128d)_mm_shuffle_epi8((__m128i)elemb2, mask);
        __m128d ae = (__m128d)_mm_or_si128((__m128i)temp,(__m128i)temp2);
        mask = _mm_set_epi8(255, 255, 255, 255, 255, 255, 255, 255,
                        15,14,13, 12, 11, 10, 9, 8);
        temp = (__m128d)_mm_shuffle_epi8((__m128i)elemb1, mask);
        mask = _mm_set_epi8(15,14,13, 12, 11, 10, 9, 8,
                        255, 255, 255, 255, 255, 255, 255, 255);
        temp2 = (__m128d)_mm_shuffle_epi8((__m128i)elemb2, mask);
        __m128d bf = (__m128d)_mm_or_si128((__m128i)temp,(__m128i)temp2);
        // multiplication
        ae = _mm_mul_pd(ae, elemA);
        bf = _mm_mul_pd(bf, elemA);
        // add to running sum
        sum[offset] = _mm_add_pd(sum[offset],ae);
        sum[offset+1] = _mm_add_pd(sum[offset+1],bf);
      }
    }
    for(int x=0;x<4;x++){
      // hadd
      __m128i mask_hadd = _mm_set_epi8(7 ,6 ,5, 4, 3, 2, 1, 0,
                        15, 14, 13, 12, 11, 10, 9, 8);
      __m128d sum_shuffled= (__m128d)_mm_shuffle_epi8((__m128i)sum[x], mask_hadd);
      sum[x] = _mm_add_pd(sum[x],sum_shuffled);
      _mm_store_sd(&res[i][x], sum[x]);
      sum[x] = _mm_setzero_pd(); // reset running sums
    }

  }

  return res;
}

fl48** matrix_matrix_mul_f48_SSE(fl48** a, fl48** b){
  fl48** res = new fl48*[4];
  for(int i=0; i<4; i++) {
    res[i] = new fl48[4];
  }
  for(int i=0; i<4; i++){
    __m128d sum[4]; // SIZE
    for(int x=0;x<4;x++){ // initialisation of final sum
      sum[x] = _mm_setzero_pd();
    }
    for(int j=0; j<4; j+=2){
      for(int offset=0; offset<4; offset+=2){
        // load mask
        __m128i load_mask = _mm_set_epi8(11, 10, 9, 8, 7, 6, 255, 255,
                       5, 4, 3, 2, 1, 0, 255, 255);
        __m128i elemA = _mm_loadu_si128((__m128i*) &a[i][j]);
        elemA = _mm_shuffle_epi8(elemA, load_mask);
        __m128i elemb1 = _mm_loadu_si128((__m128i*) &b[j][offset]);
        elemb1 = _mm_shuffle_epi8(elemb1, load_mask);
        __m128i elemb2 = _mm_loadu_si128((__m128i*) &b[j+1][offset]);
        elemb2 = _mm_shuffle_epi8(elemb2, load_mask);

        __m128i mask = _mm_set_epi8(255, 255, 255, 255, 255, 255, 255, 255,
                7 ,6 ,5, 4, 3, 2, 1, 0);
        __m128d temp = (__m128d)_mm_shuffle_epi8((__m128i)elemb1, mask);
        mask = _mm_set_epi8(7 ,6 ,5, 4, 3, 2, 1, 0,
            255, 255, 255, 255, 255, 255, 255, 255);
        __m128d temp2 = (__m128d)_mm_shuffle_epi8((__m128i)elemb2, mask);
        __m128d ae = (__m128d)_mm_or_si128((__m128i)temp,(__m128i)temp2);
        mask = _mm_set_epi8(255, 255, 255, 255, 255, 255, 255, 255,
                        15,14,13, 12, 11, 10, 9, 8);
        temp = (__m128d)_mm_shuffle_epi8((__m128i)elemb1, mask);
        mask = _mm_set_epi8(15,14,13, 12, 11, 10, 9, 8,
                        255, 255, 255, 255, 255, 255, 255, 255);
        temp2 = (__m128d)_mm_shuffle_epi8((__m128i)elemb2, mask);
        __m128d bf = (__m128d)_mm_or_si128((__m128i)temp,(__m128i)temp2);
        // multiplication
        ae = _mm_mul_pd(ae, (__m128d)elemA);
        bf = _mm_mul_pd(bf, (__m128d)elemA);
        // add to running sum
        sum[offset] = _mm_add_pd(sum[offset],ae);
        sum[offset+1] = _mm_add_pd(sum[offset+1],bf);
      }
    }
    for(int x=0;x<4;x++){
      // hadd
      __m128i mask_hadd = _mm_set_epi8(7 ,6 ,5, 4, 3, 2, 1, 0,
                        15, 14, 13, 12, 11, 10, 9, 8);
      __m128d sum_shuffled= (__m128d)_mm_shuffle_epi8((__m128i)sum[x], mask_hadd);
      sum[x] = _mm_add_pd(sum[x],sum_shuffled);
      double temp;
      _mm_store_sd(&temp, sum[x]);
      res[i][x] = fl48(temp);
      sum[x] = _mm_setzero_pd(); // reset running sums
    }

  }

  return res;
}

    /*** END LEVEL 3 ***/
