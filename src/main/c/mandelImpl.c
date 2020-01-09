#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
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
    const __m256d xmin = _mm256_set1_pd(xStart);
    const __m256d ymin = _mm256_set1_pd(yStart);
    const __m256d xscale = _mm256_set1_pd(xInc);
    const __m256d yscale = _mm256_set1_pd(yInc);
    const __m256d threshold = _mm256_set1_pd(sqrEscapeRadius);
    const __m256d one = _mm256_set1_pd(1);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < height; y++) {
        // as long as the assignment loop is failing, we calc some pixels less to avoid writing outside array limits
        for (int x = 0; x < width; x += 4) {
            __m256d mx = _mm256_set_pd(x + 3, x + 2, x + 1, x + 0);
            __m256d my = _mm256_set1_pd(y);
            __m256d cr = _mm256_add_pd(_mm256_mul_pd(mx, xscale), xmin);
            __m256d ci = _mm256_add_pd(_mm256_mul_pd(my, yscale), ymin);
            __m256d zr = cr;
            __m256d zi = ci;

            int k = 0;
            // store the iterations
            __m256d mk = _mm256_set1_pd(k);

            // last Zr/Zi values -> make them accessible as float vector
            union {
                double f[4];
                __m256d m;
            } vLastZr, vLastZi;
            vLastZr.m = _mm256_set1_pd(0);
            vLastZi.m = _mm256_set1_pd(0);

            __m256d previousInsideMask = _mm256_set1_pd(0);

            while (++k <= maxIterations) {
                /* Compute z1 from z0 */
                const __m256d zr2 = _mm256_mul_pd(zr, zr);
                const __m256d zi2 = _mm256_mul_pd(zi, zi);

                const __m256d mag2 = _mm256_add_pd(zr2, zi2);
                const __m256d insideMask = _mm256_cmp_pd(mag2, threshold, _CMP_LT_OS);
                /* Increment k for all vectors inside */
                mk = _mm256_add_pd(_mm256_and_pd(insideMask, one), mk);

                // store last inside values of z
                // copy only if inside mask changes for the vector (xor previous and current
                const __m256d noticeZMask = _mm256_xor_pd(insideMask, previousInsideMask);
                vLastZr.m = _mm256_add_pd(_mm256_and_pd(noticeZMask, zr), vLastZr.m);
                vLastZi.m = _mm256_add_pd(_mm256_and_pd(noticeZMask, zi), vLastZi.m);
                previousInsideMask = insideMask;

                /* Early bailout? */
                if (_mm256_testz_pd(insideMask, _mm256_set1_pd(-1))) {
                    break;
                }

                __m256d zrzi = _mm256_mul_pd(zr, zi);
                /* zr1 = zr0 * zr0 - zi0 * zi0 + cr */
                /* zi1 = zr0 * zi0 + zr0 * zi0 + ci */
                zr = _mm256_add_pd(_mm256_sub_pd(zr2, zi2), cr);
                zi = _mm256_add_pd(_mm256_add_pd(zrzi, zrzi), ci);
            }

            // convert counter to int and make it accessible via array index
            union {
                int i[4];
                __m128i m;
            } vCount;
            vCount.m = _mm256_cvtpd_epi32(mk);

            const int tIndex = x + y * width;
            for ( int i=0; i<4 && x+i<width; i++ ) {
                iters[tIndex+i] = vCount.i[i];
                lastZrs[tIndex] = (double)vLastZr.f[i];
                lastZis[tIndex] = (double)vLastZi.f[i];
            }

            //            const int *counts = (int *)&mCount;
            //            const double *lastZr = (double *)&mlastZr;
            //            const double *lastZi = (double *)&mlastZi;
            // unclear why this did not work with loop
            //            for ( int i=0; i<4 && x+i<width; i++ ) {
            //                iters[tIndex + i]  = counts[i];
            //                lastZrs[tIndex+i] = (double)lastZr[i];
            //                lastZis[tIndex+i] = (double)lastZi[i];
            //            }
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
    const __m256 xmin = _mm256_set1_ps(xStart);
    const __m256 ymin = _mm256_set1_ps(yStart);
    const __m256 xscale = _mm256_set1_ps(xInc);
    const __m256 yscale = _mm256_set1_ps(yInc);
    const __m256 threshold = _mm256_set1_ps(sqrEscapeRadius);
    const __m256 one = _mm256_set1_ps(1);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < height; y++) {
        // as long as the assignment loop is failing, we calc some pixels less to avoid writing outside array limits
        for (int x = 0; x < width; x += 8) {
            __m256 mx = _mm256_set_ps(x+7, x+6, x+5, x+4, x + 3, x + 2, x + 1, x + 0);
            __m256 my = _mm256_set1_ps(y);
            __m256 cr = _mm256_add_ps(_mm256_mul_ps(mx, xscale), xmin);
            __m256 ci = _mm256_add_ps(_mm256_mul_ps(my, yscale), ymin);
            __m256 zr = cr;
            __m256 zi = ci;

            int k = 0;
            // store the iterations
            __m256 mk = _mm256_set1_ps(k);

            // last Zr/Zi values -> make them accessible as float vector
            union {
                float f[8];
                __m256 m;
            } vLastZr, vLastZi;
            vLastZr.m = _mm256_set1_ps(0);
            vLastZi.m = _mm256_set1_ps(0);

            __m256 previousInsideMask = _mm256_set1_ps(0);

            while (++k <= maxIterations) {
                /* Compute z1 from z0 */
                const __m256 zr2 = _mm256_mul_ps(zr, zr);
                const __m256 zi2 = _mm256_mul_ps(zi, zi);

                const __m256 mag2 = _mm256_add_ps(zr2, zi2);
                const __m256 insideMask = _mm256_cmp_ps(mag2, threshold, _CMP_LT_OS);
                /* Increment k for all vectors inside */
                mk = _mm256_add_ps(_mm256_and_ps(insideMask, one), mk);

                // store last inside values of z
                // copy only if inside mask changes for the vector (xor previous and current
                const __m256 noticeZMask = _mm256_xor_ps(insideMask, previousInsideMask);
                vLastZr.m  = _mm256_add_ps(_mm256_and_ps(noticeZMask, zr), vLastZr.m);
                vLastZi.m  = _mm256_add_ps(_mm256_and_ps(noticeZMask, zi), vLastZi.m);
                previousInsideMask = insideMask;

                /* Early bailout? */
                if (_mm256_testz_ps(insideMask, _mm256_set1_ps(-1))) {
                    break;
                }

                __m256 zrzi = _mm256_mul_ps(zr, zi);
                /* zr1 = zr0 * zr0 - zi0 * zi0 + cr */
                /* zi1 = zr0 * zi0 + zr0 * zi0 + ci */
                zr = _mm256_add_ps(_mm256_sub_ps(zr2, zi2), cr);
                zi = _mm256_add_ps(_mm256_add_ps(zrzi, zrzi), ci);
            }

            // convert counter to int and make it accessible via array index
            union {
                int i[8];
                __m256i m;
            } vCount;
            vCount.m = _mm256_cvtps_epi32(mk);

            const int tIndex = x + y * width;
            for ( int i=0; i<8 && x+i<width; i++ ) {
                iters[tIndex+i] = vCount.i[i];
                lastZrs[tIndex] = (double)vLastZr.f[i];
                lastZis[tIndex] = (double)vLastZi.f[i];
            }


            // totally unclear why the loop did not work
            //                        int *counts = (int *)&mCount;
            //                        float *lastZr = (float *)&mlastZr;
            //                        float *lastZi = (float *)&mlastZi;
            //             for ( int i=0; i<8 && x+i<width; i++ ) {
            //                 iters[tIndex+i] = counts[i];
            //                lastZrs[tIndex+i] = (double)lastZr[i];
            //                lastZis[tIndex+i] = (double)lastZi[i];
            //             }
        }
    }
}



void
mandel_double(unsigned int *iters,
              double *lastZrs,
              double *lastZis,
              double *distancesR,
              double *distancesI,
              bool calcDistance,
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

            // distance
            double dr = 1;
            double di = 0;
            double new_dr;

            int count = 0;
            for (; count<maxIterations; count++){
                const double zrsqr = zr * zr;
                const double zisqr = zi * zi;

                if ( (zrsqr + zisqr) >= escape ) {
                    break;
                }

                if ( calcDistance ) {
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
            const int tIndex = x + y * width;
            iters[tIndex] = count;
            lastZrs[tIndex] = zr;
            lastZis[tIndex] = zi;
            if ( calcDistance ) {
                distancesR[tIndex] = dr;
                distancesI[tIndex] = di;
            }
        }
    }

}
