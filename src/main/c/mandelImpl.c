#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mandel.h"

// copied from https://github.com/skeeto/mandel-simd/blob/master/mandel_avx.c


void
mandel_avxd(unsigned int *iters,
            double *lastZrs,
            double *lastZis,
            const int width,
            const int height,
            const double xStart,
            const double yStart,
            const double xInc,
            const double yInc,
            const int maxIterations,
            const double sqrEscapeRadius)
{
    __m256d xmin = _mm256_set1_pd(xStart);
    __m256d ymin = _mm256_set1_pd(yStart);
    __m256d xscale = _mm256_set1_pd(xInc);
    __m256d yscale = _mm256_set1_pd(yInc);
    __m256d threshold = _mm256_set1_pd(sqrEscapeRadius);
    __m256d one = _mm256_set1_pd(1);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 4) {
            __m256d mx = _mm256_set_pd(x + 3, x + 2, x + 1, x + 0);
            __m256d my = _mm256_set1_pd(y);
            __m256d cr = _mm256_add_pd(_mm256_mul_pd(mx, xscale), xmin);
            __m256d ci = _mm256_add_pd(_mm256_mul_pd(my, yscale), ymin);
            __m256d zr = cr;
            __m256d zi = ci;
            int k = 1;
            __m256d mk = _mm256_set1_pd(k);
            __m256d mlastZr = _mm256_set1_pd(0);
            __m256d mlastZi = _mm256_set1_pd(0);
            __m256d copyMask;

            while (++k <= maxIterations) {
                /* Compute z1 from z0 */
                __m256d zr2 = _mm256_mul_pd(zr, zr);
                __m256d zi2 = _mm256_mul_pd(zi, zi);
                __m256d zrzi = _mm256_mul_pd(zr, zi);
                /* zr1 = zr0 * zr0 - zi0 * zi0 + cr */
                /* zi1 = zr0 * zi0 + zr0 * zi0 + ci */
                zr = _mm256_add_pd(_mm256_sub_pd(zr2, zi2), cr);
                zi = _mm256_add_pd(_mm256_add_pd(zrzi, zrzi), ci);

                /* Increment k */
                zr2 = _mm256_mul_pd(zr, zr);
                zi2 = _mm256_mul_pd(zi, zi);
                __m256d mag2 = _mm256_add_pd(zr2, zi2);
                __m256d mask = _mm256_cmp_pd(mag2, threshold, _CMP_LT_OS);
                mk = _mm256_add_pd(_mm256_and_pd(mask, one), mk);

                // TODO tut nicht
                // Werte merken, wenn threshhold erreicht ist
                // es darf immer nur dann kopiert werden, wenn die Vector position von drinnen nach draussen wechselt, also xor vorher, nachher 1
                copyMask = _mm256_xor_pd(mask, copyMask);
                mlastZr = _mm256_add_pd(_mm256_and_pd(copyMask, zr), mlastZr);
                mlastZi = _mm256_add_pd(_mm256_and_pd(copyMask, zi), mlastZi);


                /* Early bailout? */
                if (_mm256_testz_pd(mask, _mm256_set1_pd(-1)))
                    break;
            }

            __m128i mCount = _mm256_cvtpd_epi32(mk);
            int *counts = (int *)&mCount;
            double *lastZr = (double *)&mlastZr;
            double *lastZi = (double *)&mlastZi;

            const int tIndex = x + y * width;
            // unclear why this did not work with loop
            //            for ( int i=0; i<4 && x+i<width; i++ ) {
            //                iters[tIndex + i]  = counts[i];
            //                lastZrs[tIndex+i] = (double)lastZr[i];
            //                lastZis[tIndex+i] = (double)lastZi[i];
            //            }
            iters[tIndex]  = counts[0];
            iters[tIndex+1]  = counts[1];
            iters[tIndex+2]  = counts[2];
            iters[tIndex+3]  = counts[3];
            lastZrs[tIndex] = lastZr[0];
            lastZrs[tIndex+1] = lastZr[1];
            lastZrs[tIndex+2] = lastZr[2];
            lastZrs[tIndex+3] = lastZr[3];
            lastZis[tIndex] = lastZi[0];
            lastZis[tIndex+1] = lastZi[1];
            lastZis[tIndex+2] = lastZi[2];
            lastZis[tIndex+3] = lastZi[3];
        }
    }

}



void
mandel_avxs(unsigned int *iters,
            double *lastZrs,
            double *lastZis,
            const int width,
            const int height,
            const float xStart,
            const float yStart,
            const float xInc,
            const float yInc,
            const int maxIterations,
            const float sqrEscapeRadius)
{
    __m256 xmin = _mm256_set1_ps(xStart);
    __m256 ymin = _mm256_set1_ps(yStart);
    __m256 xscale = _mm256_set1_ps(xInc);
    __m256 yscale = _mm256_set1_ps(yInc);
    __m256 threshold = _mm256_set1_ps(sqrEscapeRadius);
    __m256 one = _mm256_set1_ps(1);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 8) {
            __m256 mx = _mm256_set_ps(x+7, x+6, x+5, x+4, x + 3, x + 2, x + 1, x + 0);
            __m256 my = _mm256_set1_ps(y);
            __m256 cr = _mm256_add_ps(_mm256_mul_ps(mx, xscale), xmin);
            __m256 ci = _mm256_add_ps(_mm256_mul_ps(my, yscale), ymin);
            __m256 zr = cr;
            __m256 zi = ci;
            int k = 1;
            __m256 mk = _mm256_set1_ps(k);
            __m256 mlastZr = _mm256_set1_ps(0);
            __m256 mlastZi = _mm256_set1_ps(0);
            __m256 copyMask;
            while (++k <= maxIterations) {
                /* Compute z1 from z0 */
                __m256 zr2 = _mm256_mul_ps(zr, zr);
                __m256 zi2 = _mm256_mul_ps(zi, zi);
                __m256 zrzi = _mm256_mul_ps(zr, zi);
                /* zr1 = zr0 * zr0 - zi0 * zi0 + cr */
                /* zi1 = zr0 * zi0 + zr0 * zi0 + ci */
                zr = _mm256_add_ps(_mm256_sub_ps(zr2, zi2), cr);
                zi = _mm256_add_ps(_mm256_add_ps(zrzi, zrzi), ci);

                /* Increment k */
                zr2 = _mm256_mul_ps(zr, zr);
                zi2 = _mm256_mul_ps(zi, zi);
                __m256 mag2 = _mm256_add_ps(zr2, zi2);
                __m256 mask = _mm256_cmp_ps(mag2, threshold, _CMP_LT_OS);
                mk = _mm256_add_ps(_mm256_and_ps(mask, one), mk);

                // TODO tut nicht
                // Werte merken, wenn threshold erreicht ist
                // es darf immer nur dann kopiert werden, wenn die Vector position von drinnen nach draussen wechselt, also xor vorher, nachher 1
                copyMask = _mm256_xor_ps(mask, copyMask);
                mlastZr = _mm256_add_ps(_mm256_and_ps(copyMask, zr), mlastZr);
                mlastZi = _mm256_add_ps(_mm256_and_ps(copyMask, zi), mlastZi);


                /* Early bailout? */
                if (_mm256_testz_ps(mask, _mm256_set1_ps(-1))) {
                    break;
                }
            }

            __m256i mCount = _mm256_cvtps_epi32(mk);
            int *counts = (int *)&mCount;
            float *lastZr = (float *)&mlastZr;
            float *lastZi = (float *)&mlastZi;

            const int tIndex = x + y * width;
            // totally unclear why the loop did not work
            //             for ( int i=0; i<8 && x+i<width; i++ ) {
            //                 iters[tIndex+i] = counts[i];
            //                lastZrs[tIndex+i] = (double)lastZr[i];
            //                lastZis[tIndex+i] = (double)lastZi[i];
            //             }
            iters[tIndex]  = counts[0];
            iters[tIndex+1]  = counts[1];
            iters[tIndex+2]  = counts[2];
            iters[tIndex+3]  = counts[3];
            iters[tIndex+4]  = counts[4];
            iters[tIndex+5]  = counts[5];
            iters[tIndex+6]  = counts[6];
            iters[tIndex+7]  = counts[7];
            lastZrs[tIndex] = (double)lastZr[0];
            lastZrs[tIndex+1] = (double)lastZr[1];
            lastZrs[tIndex+2] = (double)lastZr[2];
            lastZrs[tIndex+3] = (double)lastZr[3];
            lastZrs[tIndex+4] = (double)lastZr[4];
            lastZrs[tIndex+5] = (double)lastZr[5];
            lastZrs[tIndex+6] = (double)lastZr[6];
            lastZrs[tIndex+7] = (double)lastZr[7];
            lastZis[tIndex] = (double)lastZi[0];
            lastZis[tIndex+1] = (double)lastZi[1];
            lastZis[tIndex+2] = (double)lastZi[2];
            lastZis[tIndex+3] = (double)lastZi[3];
            lastZis[tIndex+4] = (double)lastZi[4];
            lastZis[tIndex+5] = (double)lastZi[5];
            lastZis[tIndex+6] = (double)lastZi[6];
            lastZis[tIndex+7] = (double)lastZi[7];

        }
    }

}



void
mandel_double(unsigned int *iters,
              double *lastZrs,
              double *lastZis,
              const int width,
              const int height,
              const double xStart,
              const double yStart,
              const double xInc,
              const double yInc,
              const int maxIterations,
              const double sqrEscapeRadius)
{
    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x ++) {
            const double cr = xStart + x*xInc;
            const double ci = yStart + y*yInc;

            const double escape = sqrEscapeRadius;

            double zr = cr;
            double zi = ci;
            double new_zr = 0.0;

            int count = 0;
            for (; count<maxIterations; count++){
                const double zrsqr = zr * zr;
                const double zisqr = zi * zi;

                if ( (zrsqr + zisqr) >= escape ) {
                    break;
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
            const int tIndex = x + y * width;
            iters[tIndex] = count;
            lastZrs[tIndex] = zr;
            lastZis[tIndex] = zi;
        }
    }

}
