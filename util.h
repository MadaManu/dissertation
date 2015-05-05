#include "stdio.h"
#include "stdlib.h"
#include "stdint.h"
#include "xmmintrin.h"
#include "limits.h"
#include "math.h"
#include "float.h"
#include "time.h"

static inline __m128 __attribute__((always_inline)) pseudo_sqrtps(__m128 val) {
  __m128 zero = (__m128) _mm_setzero_si128();
  __m128 constA = _mm_set_ps1(-3.0);
  __m128 constB = _mm_set_ps1(-0.5);
  __m128 notZero = zero;
  __m128 rsq, tmp;

  notZero = _mm_cmpneq_ps(val, notZero);
  rsq     = _mm_rsqrt_ps(val);
  rsq     = _mm_and_ps(notZero, rsq);
  val     = _mm_mul_ps(rsq, val);
  rsq     = _mm_mul_ps(val, rsq);
  val     = _mm_mul_ps(constA, val);
  rsq     = _mm_add_ps(constB, rsq);
  val     = _mm_mul_ps(rsq, val);
  return val;
}

typedef struct {
  uint64_t *bins;
  double *labels;
  int count;
} histogram;

typedef struct {
  uint64_t frequency;
  double label;
} pair;

int findIndex(double needle, double* haystack, int haystackSize) {
  for(int i = 0; i < haystackSize; i++) {
    if (haystack[i] == needle) { return i; }
  }
  return -1;
}

int findIndexThresh(double threshold, double needle, double* haystack, int haystackSize) {
  for(int i = 0; i < haystackSize; i++) {
    if (needle < (haystack[i] + threshold) &&
        needle >= haystack[i] ) { return i; }
  }
  return -1;
}

void print(histogram *h, uint64_t threshold) {
  for (int i = 0; i < h->count; i++) {
    if(h->bins[i] > threshold) {
      printf("%f\t%ld\n", h->labels[i], h->bins[i]);
    }
  }
}

histogram *hist(double* data, int count) {
  double min = 0;
  double max = 0;
  for(int i = 0; i < count; i++) {
    min = (data[i] < min) ? data[i] : min;
    max = (data[i] > max) ? data[i] : max;
  }

  int bins = (max - min) + 1;
  double *m = (double*) malloc(bins * sizeof(double));
  for(int i = 0; i < bins; i++) { m[i] = min+i; }

  uint64_t *h = (uint64_t*) malloc(bins * sizeof(uint64_t));
  for(int i = 0; i < bins; i++) { h[i] = 0; }
  for(int i = 0; i < count; i++) {
    h[findIndex(data[i], m, bins)] += 1;
  }

  histogram *histo = (histogram*) malloc(sizeof(histogram));
  histo->bins = h;
  histo->labels = m;
  histo->count = bins;
  return histo;
}

histogram *uhist(uint64_t* data, int count) {
  double min = 0;
  double max = 0;
  for(int i = 0; i < count; i++) {
    min = (((double)data[i]) < min) ? ((double)data[i]) : min;
    max = (((double)data[i]) > max) ? ((double)data[i]) : max;
  }

  int bins = (max - min) + 1;
  double *m = (double*) malloc(bins * sizeof(double));
  for(int i = 0; i < bins; i++) { m[i] = min+i; }

  uint64_t *h = (uint64_t*) malloc(bins * sizeof(uint64_t));
  for(int i = 0; i < bins; i++) { h[i] = 0; }
  for(int i = 0; i < count; i++) {
    h[findIndex(((double)data[i]), m, bins)] += 1;
  }

  histogram *histo = (histogram*) malloc(sizeof(histogram));
  histo->bins = h;
  histo->labels = m;
  histo->count = bins;
  return histo;
}

histogram *uhistThresh(uint64_t* data, int count, uint64_t threshold) {
  double min = 0;
  double max = 0;
  for(int i = 0; i < count; i++) {
    min = (((double)data[i]) < min) ? ((double)data[i]) : min;
    max = (((double)data[i]) > max) ? ((double)data[i]) : max;
  }

  uint64_t bins = (uint64_t) (((max - min) / (double) threshold) + 1.0);
  double *m = (double*) malloc(bins * sizeof(double));
  for(int i = 0; i < bins; i++) { m[i] = min+((double)i*threshold); }

  uint64_t *h = (uint64_t*) malloc(bins * sizeof(uint64_t));
  for(int i = 0; i < bins; i++) { h[i] = 0; }
  for(int i = 0; i < count; i++) {
    h[findIndexThresh(((double)threshold), ((double)data[i]), m, bins)] += 1;
  }

  histogram *histo = (histogram*) malloc(sizeof(histogram));
  histo->bins = h;
  histo->labels = m;
  histo->count = bins;
  return histo;
}

histogram *uhistRange(uint64_t* data, uint64_t dmin, int count) {
  double min = (double) dmin;
  double max = 0;
  for(int i = 0; i < count; i++) {
    min = (((double)data[i]) < min) ? ((double)data[i]) : min;
    max = (((double)data[i]) > max) ? ((double)data[i]) : max;
  }

  int bins = (max - min) + 1;
  double *m = (double*) malloc(bins * sizeof(double));
  for(int i = 0; i < bins; i++) { m[i] = min+i; }

  uint64_t *h = (uint64_t*) malloc(bins * sizeof(uint64_t));
  for(int i = 0; i < bins; i++) { h[i] = 0; }
  for(int i = 0; i < count; i++) {
    h[findIndex(((double)data[i]), m, bins)] += 1;
  }

  histogram *histo = (histogram*) malloc(sizeof(histogram));
  histo->bins = h;
  histo->labels = m;
  histo->count = bins;
  return histo;
}

double variance(double mean, int population_count, histogram *h) {
  double *devs = (double*) malloc(h->count * sizeof(double));
  double sum = 0.0;
  for (int i = 0; i < h->count; i++) { devs[i] = h->labels[i] - mean; }
  for (int i = 0; i < h->count; i++) { devs[i] = h->bins[i] * (devs[i] * devs[i]); }
  for (int i = 0; i < h->count; i++) { sum += devs[i]; }
  return (sum/((double)(population_count - 1)));
}


double weighted_average(histogram *h) {

  double sum_freqs = 0.0;
  for (int i = 0; i < h->count; i++) {
    sum_freqs += ((double) h->bins[i]);
  }

  double avg_observation = 0.0;
  for (int i = 0; i < h->count; i++) {
    avg_observation += ((double) h->bins[i]) * h->labels[i];
  }
  avg_observation /= sum_freqs;
  return avg_observation;

}

double min_label(histogram *h) {
  double theMin = DBL_MAX;
  for (int i = 0; i < h->count; i++) {
    if (h->labels[i] < theMin && h->bins[i] > 0) { theMin = h->labels[i]; }
  }
  return theMin;
}

double max_label(histogram *h) {
  double theMax = 0.0;
  for (int i = 0; i < h->count; i++) {
    if (h->labels[i] > theMax && h->bins[i] > 0) { theMax = h->labels[i]; }
  }
  return theMax;
}

double most_common_label(histogram *h) {
  uint64_t maxfreq = 0;
  double theLabel;
  for (int i = 0; i < h->count; i++) {
    if (h->bins[i] > maxfreq) { theLabel = h->labels[i]; maxfreq = h->bins[i]; }
  }
  return theLabel;
}

double most_similar_label(double label, histogram *h) {
  double difference = DBL_MAX;
  double found;
  for (int i = 0; i < h->count; i++) {
    if ((fabs(h->labels[i] - label) < difference) && h->bins[i] > 0) {
      difference = fabs(h->labels[i] - label);
      found = h->labels[i];
    }
  }
  return found;
}

double median_label(histogram *h, int population_count) {
  uint64_t sum_of_freqs = 0;
  int half_popcount = (population_count + 1) / 2;

  for (int i = 0; i < h->count; i++) {
    sum_of_freqs += h->bins[i];

    if (sum_of_freqs > half_popcount) {

      return ((double) h->labels[i]);

    } else if (sum_of_freqs == half_popcount ) {

      if ((population_count % 2 == 0)) {
        return ((double) h->labels[i]);
      } else {
        return ((double) (h->labels[i] + h->labels[i+1])) / 2.0;
      }
    }
  }

  return (-1.0);
}

uint64_t query(double label, histogram *h) {
  return h->bins[findIndex(label, h->labels, h->count)];
}

static inline uint64_t __attribute__((always_inline)) clocks() {
    uint64_t a, d;
    volatile uint64_t x, y;

    __asm__ volatile ( "cpuid"
                     : "=a" (y)
                     : "0" (x)
                     : "%rbx", "%rcx", "%rdx", "memory");

    __asm__ volatile ("rdtsc"
                     : "=a" (a), "=d" (d)
                     :
                     : "memory");

    return (a | d << 32);
}

static inline uint64_t __attribute__((always_inline)) clocks_noserialize() {
    uint64_t a, d;

    __asm__  ("rdtsc"
             : "=a" (a), "=d" (d)
             : );

    return (a | d << 32);
}
