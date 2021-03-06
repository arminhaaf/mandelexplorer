//
// Created by armin on 10.01.20.
//

#ifndef MANDELAVXDD_MANDELSSEDD_H
#define MANDELAVXDD_MANDELSSEDD_H

#include <immintrin.h>
#include <stdbool.h>
#include <stdint.h>


#define MODE_MANDEL 1
#define MODE_MANDEL_DISTANCE 2
#define MODE_JULIA 3

typedef struct {
    __m128d hi;
    __m128d lo;
} DD2;


void mandel_ssedd(
        int32_t *iters,
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


#endif //MANDELAVXDD_MANDELAVXDD_H
