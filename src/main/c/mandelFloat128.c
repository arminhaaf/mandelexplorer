#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "mandel.h"

void
mandel_longdouble(int32_t *iters,
              double *lastZrs,
              double *lastZis,
              double *distancesR,
              double *distancesI,
              const int32_t mode,
              const int32_t width,
              const int32_t height,
          const double pxStartHi,
          const double pxStartLo,
          const double pyStartHi,
          const double pyStartLo,
          const double pjuliaCrHi,
          const double pjuliaCrLo,
          const double pjuliaCiHi,
          const double pjuliaCiLo,
          const double pxIncHi,
          const double pxIncLo,
          const double pyIncHi,
          const double pyIncLo,
              const int32_t maxIterations,
              const double sqrEscapeRadius)
{
    __float128 xStart = pxStartHi + pxStartLo;
    __float128 yStart = pyStartHi + pyStartLo;
    __float128 juliaCr = pjuliaCrHi + pjuliaCrLo;
    __float128 juliaCi = pjuliaCiHi + pjuliaCiLo;
    __float128 xInc = pxIncHi + pxIncLo;
    __float128 yInc = pyIncHi + pyIncLo;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < height; y++) {
        const __float128 tY = yStart + y*yInc;
        const __float128 ci = mode == MODE_JULIA ? juliaCi : tY;

        for (int x = 0; x < width; x ++) {
            const __float128 tX = xStart + x*xInc;
            const __float128 cr = mode == MODE_JULIA ? juliaCr : tX;

            __float128 zr = tX;
            __float128 zi = tY;
            __float128 new_zr = 0.0;

            // distance
            __float128 dr = 1;
            __float128 di = 0;
            __float128 new_dr;

            int32_t count = 0;
            for (; count<maxIterations; count++){
                const __float128 zrsqr = zr * zr;
                const __float128 zisqr = zi * zi;

                if ( (zrsqr + zisqr) >= sqrEscapeRadius ) {
                    break;
                }

                if ( mode==MODE_MANDEL_DISTANCE ) {
                    new_dr = 2.0 * (zr * dr - zi * di) + 1.0;
                    di = 2.0 * (zr * di + zi * dr);
                    dr = new_dr;
                }

                new_zr = (zrsqr - zisqr) + cr;
                zi = ((2.0f * zr) * zi) + ci;
                zr = new_zr;

                //If in a periodic orbit, assume it is trapped
                if (zr == 0.0 && zi == 0.0) {
                    count = maxIterations;
                    break;
                }
            }
            const int32_t tIndex = x + y * width;
            iters[tIndex] = count;
            lastZrs[tIndex] = zr;
            lastZis[tIndex] = zi;
            if ( mode==MODE_MANDEL_DISTANCE ) {
                distancesR[tIndex] = dr;
                distancesI[tIndex] = di;
            }
        }
    }
}
