#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "mandel.h"

// copied from https://github.com/skeeto/mandel-simd/blob/master/mandel_avx.c
// changed a lot to make it easier readable and support distance and last z



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
              const double sqrEscapeRadius)
{
    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < height; y++) {
        const double tY = yStart + y*yInc;
        const double ci = mode == MODE_JULIA ? juliaCi : tY;

        for (int x = 0; x < width; x ++) {
            const double tX = xStart + x*xInc;
            const double cr = mode == MODE_JULIA ? juliaCr : tX;

            double zr = tX;
            double zi = tY;
            double new_zr = 0.0;

            // distance
            double dr = 1;
            double di = 0;
            double new_dr;

            int32_t count = 0;
            for (; count<maxIterations; count++){
                const double zrsqr = zr * zr;
                const double zisqr = zi * zi;

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
