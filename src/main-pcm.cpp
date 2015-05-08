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
#include <clocks.h>
#include <cpucounters.h>

#include <config.h>
#include <L1.hpp>
#include <L2.hpp>
#include <L3.hpp>
#include <test.hpp>
#include <reports.hpp>

using namespace std;

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
    start = clocks();
    matrix_vector_mul_double(matrix,a);
    stop = clocks();
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
    start = clocks();
    matrix_vector_mul_f48(matrix,a);
    stop = clocks();
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
    start = clocks();
    matrix_vector_mul_SSE_double(matrix,a);
    stop = clocks();
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
    start = clocks();
    matrix_vector_mul_SSE_f48(matrix,a);
    stop = clocks();
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
    start = clocks();
    matrix_vector_mul_SSE_double_v2(matrix,a);
    stop = clocks();
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
    start = clocks();
    matrix_vector_mul_SSE_f48_v2(matrix,a);
    stop = clocks();
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
    start = clocks();
    matrix_vector_mul_SSE_f48_loop_unrolled(matrix,a);
    stop = clocks();
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
// matrix_matrix_mul_f48_SSE
// matrix_matrix_mul_double_SSE

void test_matrix_matrix_mul_double_nonSSE(){
  cout<<"DOUBLE(nonSSE) matrix-matrix multiplication..." << endl;
  double** matrixa = new double*[size];
  double** matrixb = new double*[size];
  populate_matrix(matrixa);
  populate_matrix(matrixb);

  u64 start;
  u64 stop;
  u64 diff;

  // preparing file
  ofstream outputfile;
  char path[256];
  sprintf(path, "results/level3/%d/matrix_matrix_mul_double_nonSSE.txt", size);
  outputfile.open (path);

  for ( int i = 0; i < runs; i++ ) {
    start = clocks();
    matrix_matrix_mul_double(matrixa, matrixb);
    stop = clocks();
    diff = stop - start;
    outputfile<<diff<<endl;
    // repopulate
    populate_matrix(matrixa);
    populate_matrix(matrixb);
  }
  outputfile.close();
  cout<<"Done. Results in results/level3/"<<size<<"/matrix_matrix_mul_double_nonSSE.txt"<<endl;
}

void test_matrix_matrix_mul_f48_nonSSE(){
  cout<<"F48(nonSSE) matrix-matrix multiplication..." << endl;
  f48** matrixa = new f48*[size];
  f48** matrixb = new f48*[size];
  populate_matrix(matrixa);
  populate_matrix(matrixb);

  u64 start;
  u64 stop;
  u64 diff;

  // preparing file
  ofstream outputfile;
  char path[256];
  sprintf(path, "results/level3/%d/matrix_matrix_mul_f48_nonSSE.txt", size);
  outputfile.open (path);

  for ( int i = 0; i < runs; i++ ) {
    start = clocks();
    matrix_matrix_mul_f48(matrixa, matrixb);
    stop = clocks();
    diff = stop - start;
    outputfile<<diff<<endl;
    // repopulate
    populate_matrix(matrixa);
    populate_matrix(matrixb);
  }
  outputfile.close();
  cout<<"Done. Results in results/level3/"<<size<<"/matrix_matrix_mul_f48_nonSSE.txt"<<endl;
}

void test_matrix_matrix_mul_double_SSE(){
  cout<<"DOUBLE(SSE) matrix-matrix multiplication..." << endl;
  double** matrixa = new double*[size];
  double** matrixb = new double*[size];
  populate_matrix(matrixa);
  populate_matrix(matrixb);

  u64 start;
  u64 stop;
  u64 diff;

  // preparing file
  ofstream outputfile;
  char path[256];
  sprintf(path, "results/level3/%d/matrix_matrix_mul_double_SSE.txt", size);
  outputfile.open (path);

  for ( int i = 0; i < runs; i++ ) {
    start = clocks();
    matrix_matrix_mul_double_SSE(matrixa, matrixb);
    stop = clocks();
    diff = stop - start;
    outputfile<<diff<<endl;
    // repopulate
    populate_matrix(matrixa);
    populate_matrix(matrixb);
  }
  outputfile.close();
  cout<<"Done. Results in results/level3/"<<size<<"/matrix_matrix_mul_double_SSE.txt"<<endl;
}

void test_matrix_matrix_mul_f48_SSE(){
  cout<<"F48(SSE) matrix-matrix multiplication..." << endl;
  f48** matrixa = new f48*[size];
  f48** matrixb = new f48*[size];
  populate_matrix(matrixa);
  populate_matrix(matrixb);

  u64 start;
  u64 stop;
  u64 diff;

  // preparing file
  ofstream outputfile;
  char path[256];
  sprintf(path, "results/level3/%d/matrix_matrix_mul_f48_SSE.txt", size);
  outputfile.open (path);

  for ( int i = 0; i < runs; i++ ) {
    start = clocks();
    matrix_matrix_mul_f48_SSE(matrixa, matrixb);
    stop = clocks();
    diff = stop - start;
    outputfile<<diff<<endl;
    // repopulate
    populate_matrix(matrixa);
    populate_matrix(matrixb);
  }
  outputfile.close();
  cout<<"Done. Results in results/level3/"<<size<<"/matrix_matrix_mul_f48_SSE.txt"<<endl;
}


/************ END TESTS ************/

/************* MAIN ***********************/

int main()
{
  double dummyDouble;
  f48 dummyf48;

  // dot product
  test_2ptr(dummyDouble,       "DOUBLE(SSE) dot product...", "results/level1/256", "dot_prod_SSE_double", "dot_prod_SSE_double_pcm", &dot_product_SSE_double);
  test_2ptr(dummyf48,          "f48(SSE) dot product...", "results/level1/256", "dot_prod_SSE_f48", "dot_prod_SSE_f48_pcm", &dot_product_SSE_f48);
  // scale
  test_ptr_scalar(dummyDouble, "DOUBLE(SSE) scale vector...", "results/level1/256", "scale_SSE_double", "scale_SSE_double_pcm", &scale_double_vector_SSE);
  test_ptr_scalar(dummyf48,    "f48(SSE) scale vector...", "results/level1/256", "scale_SSE_f48", "scale_SSE_f48_pcm", &scale_f48_vector_SSE);
  // max
  test_ptr(dummyDouble,        "DOUBLE(SSE) max ...", "results/level1/256", "max_SSE_double", "max_SSE_double_pcm", &absolute_max_SSE_double);
  test_ptr(dummyf48,           "f48(SSE) max ...", "results/level1/256", "max_SSE_f48", "max_SSE_f48_pcm", &absolute_max_SSE_f48);
  // min
  test_ptr(dummyDouble,        "DOUBLE(SSE) min ...", "results/level1/256", "min_SSE_double", "min_SSE_double_pcm", &absolute_min_SSE_double);
  test_ptr(dummyf48,           "f48(SSE) min ...", "results/level1/256", "min_SSE_f48", "min_SSE_f48_pcm", &absolute_min_SSE_f48);
  // magnitude
  test_ptr(dummyDouble,        "DOUBLE(SSE) magnitude ...", "results/level1/256", "magnitude_SSE_double", "magnitude_SSE_double_pcm", &magnitude_SSE_double);
  test_ptr(dummyf48,           "f48(SSE) magnitude ...", "results/level1/256", "magnitude_SSE_f48", "magnitude_SSE_f48_pcm", &magnitude_SSE_f48);
  return 0;
}
