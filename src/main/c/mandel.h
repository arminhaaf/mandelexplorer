#pragma once

#include <stdbool.h>
#include <stdint.h>

#define MODE_MANDEL 1
#define MODE_MANDEL_DISTANCE 2
#define MODE_JULIA 3

typedef struct {
    double hi;
    double lo;
} DD;


void
mandel_avxd(int32_t *iters,
            double *lastZrs,
            double *lastZis,
            double *distancesR,
            double *distancesI,
            const int32_t mode,
            const int32_t width,
            const int32_t height,
            const double xStart,
            const double yStart,
            const double juliaCr,
            const double juliaCi,
            const double xInc,
            const double yInc,
            const int32_t maxIterations,
            const double sqrEscapeRadius);

void
mandel_avxs(int32_t *iters,
            double *lastZrs,
            double *lastZis,
            double *distancesR,
            double *distancesI,
            const int32_t mode,
            const int32_t width,
            const int32_t height,
            const float xStart,
            const float yStart,
            const float juliaCr,
            const float juliaCi,
            const float xInc,
            const float yInc,
            const int32_t maxIterations,
            const float sqrEscapeRadius);


void
mandel_double(int32_t *iters,
              double *lastZrs,
              double *lastZis,
              double *distancesR,
              double *distancesI,
              const int32_t mode,
              const int32_t width,
              const int32_t height,
              const double xStart,
              const double yStart,
              const double juliaCr,
              const double juliaCi,
              const double xInc,
              const double yInc,
              const int32_t maxIterations,
              const double sqrEscapeRadius);

void
mandel_dd(int32_t *iters,
          double *lastZrs,
          double *lastZis,
          double *distancesR,
          double *distancesI,
          const int32_t mode,
          const int32_t width,
          const int32_t height,
          const double xStartHi,
          const double xStartLo,
          const double yStartHi,
          const double yStartLo,
          const double juliaCrHi,
          const double juliaCrLo,
          const double juliaCiHi,
          const double juliaCiLo,
          const double xIncHi,
          const double xIncLo,
          const double yIncHi,
          const double yIncLo,
          const int32_t maxIterations,
          const double sqrEscapeRadius);

