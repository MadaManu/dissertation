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

int main()
{
  double dummyDouble;
  f48 dummyf48;
  sprintf(sizestr, "%d", size);
  string resultspath = "results";
  string l1path = resultspath + "/level1";
  string l2path = resultspath + "/level2";
  string l3path = resultspath + "/level3";

  // LEVEL 1
  // dot product
  test_2ptr(dummyDouble,       "DOUBLE(SSE) dot product...", l1path+"/"+sizestr, "dot-prod-SSE-double", "dot-prod-SSE-double-pcm", &dot_product_SSE_double);
  test_2ptr(dummyf48,          "f48(SSE) dot product...", l1path+"/"+sizestr, "dot-prod-SSE-f48", "dot-prod-SSE-f48-pcm", &dot_product_SSE_f48);
  // scale
  test_ptr_scalar(dummyDouble, "DOUBLE(SSE) scale vector...", l1path+"/"+sizestr, "scale-SSE-double", "scale-SSE-double-pcm", &scale_double_vector_SSE);
  test_ptr_scalar(dummyf48,    "f48(SSE) scale vector...", l1path+"/"+sizestr, "scale-SSE-f48", "scale-SSE-f48-pcm", &scale_f48_vector_SSE);
  // max
  test_ptr(dummyDouble,        "DOUBLE(SSE) max ...", l1path+"/"+sizestr, "max-SSE-double", "max-SSE-double-pcm", &absolute_max_SSE_double);
  test_ptr(dummyf48,           "f48(SSE) max ...", l1path+"/"+sizestr, "max-SSE-f48", "max-SSE-f48-pcm", &absolute_max_SSE_f48);
  // min
  test_ptr(dummyDouble,        "DOUBLE(SSE) min ...", l1path+"/"+sizestr, "min-SSE-double", "min-SSE-double-pcm", &absolute_min_SSE_double);
  test_ptr(dummyf48,           "f48(SSE) min ...", l1path+"/"+sizestr, "min-SSE-f48", "min-SSE-f48-pcm", &absolute_min_SSE_f48);
  // magnitude
  test_ptr(dummyDouble,        "DOUBLE(SSE) magnitude ...", l1path+"/"+sizestr, "magnitude-SSE-double", "magnitude-SSE-double-pcm", &magnitude_SSE_double);
  test_ptr(dummyf48,           "f48(SSE) magnitude ...", l1path+"/"+sizestr, "magnitude-SSE-f48", "magnitude-SSE-f48-pcm", &magnitude_SSE_f48);

  //LEVEL 2
  // matrix-vector mul non-SSE
  test_mat_vec(dummyDouble,    "DOUBLE matrix_vector_mul ...", l2path+"/"+sizestr, "matrix-vector-mul-double", "matrix-vector-mul-double-pcm", &matrix_vector_mul_double);
  test_mat_vec(dummyf48,       "f48 matrix_vector_mul ...",    l2path+"/"+sizestr, "matrix-vector-mul-f48",    "matrix-vector-mul-f48-pcm",    &matrix_vector_mul_f48);
  // matrix-vector mul SSE v1
  test_mat_vec(dummyDouble,    "DOUBLE(SSE) matrix_vector_mul ...", l2path+"/"+sizestr, "matrix-vector-mul-SSE-double", "matrix-vector-mul-SSE-double-pcm", &matrix_vector_mul_SSE_double);
  test_mat_vec(dummyf48,       "f48(SSE) matrix_vector_mul ...",    l2path+"/"+sizestr, "matrix-vector-mul-SSE-f48",    "matrix-vector-mul-SSE-f48-pcm",    &matrix_vector_mul_SSE_f48);
  // matrix-vector mul SSE v2
  test_mat_vec(dummyDouble,    "DOUBLE(SSE v2) matrix_vector_mul ...", l2path+"/"+sizestr, "matrix-vector-mul-SSE-double-v2", "matrix-vector-mul-SSE-double-v2-pcm", &matrix_vector_mul_SSE_double_v2);
  test_mat_vec(dummyf48,       "f48(SSE v2) matrix_vector_mul ...",    l2path+"/"+sizestr, "matrix-vector-mul-SSE-f48-v2",    "matrix-vector-mul-SSE-f48-v2-pcm",    &matrix_vector_mul_SSE_f48_v2);
  // matrix-vector mul SSE unrolled
  test_mat_vec(dummyf48,       "f48(SSE unrolled) matrix_vector_mul ...",    l2path+"/"+sizestr, "matrix-vector-mul-SSE-f48-unrolled",    "matrix-vector-mul-SSE-f48-unrolled-pcm",    &matrix_vector_mul_SSE_f48_loop_unrolled);

  //LEVEL 3
  //matrix-matrix mul non-SSE
  test_mat_mat(dummyDouble,    "DOUBLE matrix_matrix_mul ...", l3path+"/"+sizestr, "matrix-matrix-mul-double", "matrix-matrix-mul-double-pcm", &matrix_matrix_mul_double);
  test_mat_mat(dummyf48,       "f48 matrix_matrix_mul ...",    l3path+"/"+sizestr, "matrix-matrix-mul-f48",    "matrix-matrix-mul-f48-pcm",    &matrix_matrix_mul_f48);
  //matrix-matrix mul non-SSE
  test_mat_mat(dummyDouble,    "DOUBLE(SSE) matrix_matrix_mul ...", l3path+"/"+sizestr, "matrix-matrix-mul-SSE-double", "matrix-matrix-mul-SSE-double-pcm", &matrix_matrix_mul_double_SSE);
  test_mat_mat(dummyf48,       "f48(SSE) matrix_matrix_mul ...",    l3path+"/"+sizestr, "matrix-matrix-mul-SSE-f48",    "matrix-matrix-mul-SSE-f48-pcm",    &matrix_matrix_mul_f48_SSE);

  return 0;
}
