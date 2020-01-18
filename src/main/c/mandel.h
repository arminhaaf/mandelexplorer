#pragma once

#include <stdbool.h>

#define MODE_MANDEL 1
#define MODE_MANDEL_DISTANCE 2
#define MODE_JULIA 3

typedef struct {
    double hi;
    double lo;
} DD;


void
mandel_avxd(unsigned int *iters,
            double *lastZrs,
            double *lastZis,
            double *distancesR,
            double *distancesI,
            int mode,
            const int width,
            const int height,
            const double xStart,
            const double yStart,
            const double juliaCr,
            const double juliaCi,
            const double xInc,
            const double yInc,
            const int maxIterations,
            const double sqrEscapeRadius);

void
mandel_avxs(unsigned int *iters,
            double *lastZrs,
            double *lastZis,
            double *distancesR,
            double *distancesI,
            int mode,
            const int width,
            const int height,
            const float xStart,
            const float yStart,
            const float juliaCr,
            const float juliaCi,
            const float xInc,
            const float yInc,
            const int maxIterations,
            const float sqrEscapeRadius);


void
mandel_double(unsigned int *iters,
              double *lastZrs,
              double *lastZis,
              double *distancesR,
              double *distancesI,
              int mode,
              const int width,
              const int height,
              const double xStart,
              const double yStart,
              const double juliaCr,
              const double juliaCi,
              const double xInc,
              const double yInc,
              const int maxIterations,
              const double sqrEscapeRadius);

void mandel_dd(
        unsigned int *iters,
        double *lastZrs,
        double *lastZis,
        double *distancesR,
        double *distancesI,
        int mode,
        const int width,
        const int height,
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
        const unsigned int maxIterations,
        const double sqrEscapeRadius);

