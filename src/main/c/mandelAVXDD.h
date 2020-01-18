//
// Created by armin on 10.01.20.
//

#ifndef MANDELAVXDD_MANDELAVXDD_H
#define MANDELAVXDD_MANDELAVXDD_H

#include <immintrin.h>
#include <stdbool.h>

#define MODE_MANDEL 1
#define MODE_MANDEL_DISTANCE 2
#define MODE_JULIA 3

typedef struct {
    __m256d hi;
    __m256d lo;
} DD4;


void mandel_avxdd(
        unsigned int *iters,
        double *lastZrs,
        double *lastZis,
        double *distancesR,
        double *distancesI,
        const int mode,
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


#endif //MANDELAVXDD_MANDELAVXDD_H
