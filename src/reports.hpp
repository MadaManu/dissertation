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
  sprintf(path, "results/level1/%d/dot_prod_SSE_double.txt", SIZE);
  ifstream test_double_dot_prod (path);
  int i=0;// counter
  if (test_double_dot_prod.is_open()) {
    while(getline(test_double_dot_prod, line)){
      double_dot_prod[i] = line;
      i++;
    }
    test_double_dot_prod.close();
  } else cout<<"ERROR: Unable to open file (results/level1/"<<SIZE<<"/dot_prod_SSE_double.txt)";
  char path2[256];
  sprintf(path2, "results/level1/%d/scale_SSE_double.txt", SIZE);
  ifstream test_double_scale (path2);
  i=0;// counter
  if (test_double_scale.is_open()) {
    while(getline(test_double_scale, line)){
      double_scale[i] = line;
      i++;
    }
    test_double_scale.close();
  } else cout<<"ERROR: Unable to open file (results/level1/"<<SIZE<<"/scale_SSE_double.txt)";
  char path3[256];
  sprintf(path3, "results/level1/%d/max_SSE_double.txt", SIZE);
  ifstream test_double_max (path3);
  i=0;// counter
  if (test_double_max.is_open()) {
    while(getline(test_double_max, line)){
      double_max[i] = line;
      i++;
    }
    test_double_max.close();
  } else cout<<"ERROR: Unable to open file (results/level1/"<<SIZE<<"/max_SSE_double.txt)";
  char path4[256];
  sprintf(path4, "results/level1/%d/min_SSE_double.txt", SIZE);
  ifstream test_double_min (path4);
  i=0;// counter
  if (test_double_min.is_open()) {
    while(getline(test_double_min, line)){
      double_min[i] = line;
      i++;
    }
    test_double_min.close();
  } else cout<<"ERROR: Unable to open file (results/level1/"<<SIZE<<"/min_SSE_double.txt)";
  char path5[256];
  sprintf(path5, "results/level1/%d/magnitude_SSE_double.txt", SIZE);
  ifstream test_double_magnitude (path5);
  i=0;// counter
  if (test_double_magnitude.is_open()) {
    while(getline(test_double_magnitude, line)){
      double_magnitude[i] = line;
      i++;
    }
    test_double_magnitude.close();
  } else cout<<"ERROR: Unable to open file (results/level1/"<<SIZE<<"/magnitude_SSE_double.txt)";
  char path6[256];
  sprintf(path6, "results/level1/%d/dot_prod_SSE_f48.txt", SIZE);
  ifstream test_f48_dot_prod (path6);
  i=0;// counter
  if (test_f48_dot_prod.is_open()) {
    while(getline(test_f48_dot_prod, line)){
      f48_dot_prod[i] = line;
      i++;
    }
    test_f48_dot_prod.close();
  } else cout<<"ERROR: Unable to open file (results/level1/"<<SIZE<<"/dot_prod_SSE_f48.txt)";
  char path7[256];
  sprintf(path7, "results/level1/%d/scale_SSE_f48.txt", SIZE);
  ifstream test_f48_scale (path7);
  i=0;// counter
  if (test_f48_scale.is_open()) {
    while(getline(test_f48_scale, line)){
      f48_scale[i] = line;
      i++;
    }
    test_f48_scale.close();
  } else cout<<"ERROR: Unable to open file (results/level1/"<<SIZE<<"/scale_SSE_f48.txt)";
  char path8[256];
  sprintf(path8, "results/level1/%d/max_SSE_f48.txt", SIZE);
  ifstream test_f48_max (path8);
  i=0;// counter
  if (test_f48_max.is_open()) {
    while(getline(test_f48_max, line)){
      f48_max[i] = line;
      i++;
    }
    test_f48_max.close();
  } else cout<<"ERROR: Unable to open file (results/level1/"<<SIZE<<"/max_SSE_f48.txt)";
  char path9[256];
  sprintf(path9, "results/level1/%d/min_SSE_f48.txt", SIZE);
  ifstream test_f48_min (path9);
  i=0;// counter
  if (test_f48_min.is_open()) {
    while(getline(test_f48_min, line)){
      f48_min[i] = line;
      i++;
    }
    test_f48_min.close();
  } else cout<<"ERROR: Unable to open file (results/level1/"<<SIZE<<"/min_SSE_f48.txt)";
  char path10[256];
  sprintf(path10, "results/level1/%d/magnitude_SSE_f48.txt", SIZE);
  ifstream test_f48_magnitude (path10);
  i=0;// counter
  if (test_f48_magnitude.is_open()) {
    while(getline(test_f48_magnitude, line)){
      f48_magnitude[i] = line;
      i++;
    }
    test_f48_magnitude.close();
  } else cout<<"ERROR: Unable to open file (results/level1/"<<SIZE<<"/magnitude_SSE_f48.txt)";
// all results loaded into memory - require structure of final report
// report for dot prod with magnitude
  ofstream dot_prod_mag_report;
  sprintf(path, "results/level1/reports/%d/dot_product_and_magnitude.csv", SIZE);
  dot_prod_mag_report.open (path);
  dot_prod_mag_report<<"Dot product (SIZE:"<<SIZE<<"),,Magnitude"<<endl;
  dot_prod_mag_report<<"Double,F48,Double,F48"<<endl;
  for(i=0; i<runs; i++){
    dot_prod_mag_report<<double_dot_prod[i]<<","<<f48_dot_prod[i]<<","<<double_magnitude[i]<<","<<f48_magnitude[i]<<endl;
  }
  dot_prod_mag_report.close();
  cout<<"Dot-product and magnitude report built sucessfully. (results/level1/reports/"<<SIZE<<"/dot_product_and_magnitude.csv)"<<endl;
// report for max with min
  ofstream max_min_report;
  sprintf(path2, "results/level1/reports/%d/maximum_and_minimum.csv", SIZE);
  max_min_report.open (path2);
  max_min_report<<"Absolute Maximum(SIZE:"<<SIZE<<"),,Absolute Minimum"<<endl;
  max_min_report<<"Double,F48,Double,F48"<<endl;
  for(i=0; i<runs; i++){
    max_min_report<<double_max[i]<<","<<f48_max[i]<<","<<double_min[i]<<","<<f48_min[i]<<endl;
  }
  max_min_report.close();
  cout<<"Absolute maximum and absolute minimum report built sucessfully. (results/level1/reports/"<<SIZE<<"/maximum_and_minimum.csv)"<<endl;
// report for scale - one of the implemented BLAS L1 functions that makes changes to the vector itself
  ofstream scale_report;
  sprintf(path3, "results/level1/reports/%d/scale.csv", SIZE);
  scale_report.open (path3);
  scale_report<<"Scale(SIZE:"<<SIZE<<")"<<endl;
  scale_report<<"Double,F48,F48-Double"<<endl;
  for(i=0; i<runs; i++){
    scale_report<<double_scale[i]<<","<<f48_scale[i]<<endl;
  }
  scale_report.close();
  cout<<"Scale report built sucessfully. (results/level1/reports/"<<SIZE<<"/scale.csv)"<<endl;

  cout<<"BLAS level 1 reports completed. (results/level1/reports/"<<SIZE<<"/)"<<endl;
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
  sprintf(path, "results/level2/%d/matrix_vector_mul_double_nonSSE.txt", SIZE);
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
  } else cout << "ERROR: Unable to open file (results/level2/"<<SIZE<<"/matrix_vector_mul_double_nonSSE.txt)";
  char path2[256];
  sprintf(path2, "results/level2/%d/matrix_vector_mul_double_v1.txt", SIZE);
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
  } else cout << "ERROR: Unable to open file (results/level2/"<<SIZE<<"/matrix_vector_mul_double_v1.txt)";
  sprintf(path2, "results/level2/%d/matrix_vector_mul_double_v2.txt", SIZE);
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
  } else cout << "ERROR: Unable to open file (results/level2/"<<SIZE<<"/matrix_vector_mul_double_v2.txt)";
    char path3[256];
  sprintf(path3, "results/level2/%d/matrix_vector_mul_f48_nonSSE.txt", SIZE);
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
  } else cout << "ERROR: Unable to open file (results/level2/"<<SIZE<<"/matrix_vector_mul_f48_nonSSE.txt)";
  char path4[256];
  sprintf(path4, "results/level2/%d/matrix_vector_mul_f48_v1.txt", SIZE);
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
  } else cout << "ERROR: Unable to open file (results/level2/"<<SIZE<<"/matrix_vector_mul_f48_v1.txt)";
  char path5[256];
  sprintf(path5, "results/level2/%d/matrix_vector_mul_f48_v2.txt", SIZE);
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
  } else cout << "ERROR: Unable to open file (results/level2/"<<SIZE<<"/matrix_vector_mul_f48_v2.txt)";
  char path6[256];
  sprintf(path6, "results/level2/%d/matrix_vector_mul_f48_loopunroll.txt", SIZE);
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
  } else cout << "ERROR: Unable to open file (results/level2/"<<SIZE<<"/matrix_vector_mul_f48_loopunroll.txt)";



  ofstream myfile;
  sprintf(path, "results/level2/reports/%d/matrix_vector_mul_report.csv", SIZE);
  myfile.open (path);

  myfile<<"Matrix-vector Multiplication (SIZE:"<<SIZE<<"),,,,,,"<<endl;
  myfile<<"Double,,,F48,,,"<<endl;
  myfile<<"non-SSE,SSE-v1,SSE-v2,non-SSE,SSE-v1,SSE-v2,SSE-loop un-rolled"<<endl;
  // TODO: refactor to new structure ^^
  for(i=0;i<runs;i++){
    myfile<<double_results_nonSSE[i]<<","<<double_results_SSEv1[i]<<","<<double_results_SSEv2[i]<<","<<f48_results_nonSSE[i]<<","<<f48_results_SSEv1[i]<<","<<f48_results_SSEv2[i]<<","<<f48_results_SSE_loopunroll[i]<<endl;
  }
  myfile.close();
  cout<<"Matrix-vector multiplication results compiled for non-SSE,SSEv1,SSEv2. (results/level2/reports/"<<SIZE<<"/matrix_vector_mul_report.txt)"<<endl;
}

    /*** LEVEL 3 REPORTS ***/
  // TODO NEEDS ADDITION AFTER IMPLEMENTATION AND TESTS

/************ END REPORTS ************/
