#include "stdio.h"
#include "string.h"
#include "stdint.h"
#include "util.h"

uint64_t diffs[TIMES];

int main(int argc, char** argv) {

  histogram *opt_h;
  FILE *myFile;

  myFile = fopen(argv[1], "r");

  for (int i = 0; i < TIMES-1; i++)
  {
      fscanf(myFile, "%llu", &diffs[i]);
  }

  opt_h = uhistThresh(&(diffs[0]), TIMES, 100000);

  printf("%f\t%f\t%f\t%f\n", max_label(opt_h)
                           , weighted_average(opt_h)
                           , median_label(opt_h, TIMES)
                           , variance(weighted_average(opt_h), TIMES, opt_h)
                           );

  free(opt_h->labels);
  free(opt_h->bins);
  free(opt_h);

  fflush(stdout);

  return 0;
}
