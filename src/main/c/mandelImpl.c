#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "mandel.h"

// copied from https://github.com/skeeto/mandel-simd/blob/master/mandel_avx.c
// changed a lot to make it easier readable and support distance and last z


void
mandel_avxd(unsigned int *iters,
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
    const __m256d xmin = _mm256_set1_pd(xStart);
    const __m256d ymin = _mm256_set1_pd(yStart);
    const __m256d xscale = _mm256_set1_pd(xInc);
    const __m256d yscale = _mm256_set1_pd(yInc);
    const __m256d threshold = _mm256_set1_pd(sqrEscapeRadius);
    const __m256d one = _mm256_set1_pd(1);
    const __m256d zero = _mm256_set1_pd(0);
    const __m256d oneminus = _mm256_set1_pd(-1);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < height; y++) {
        // as long as the assignment loop is failing, we calc some pixels less to avoid writing outside array limits
        const __m256d my = _mm256_set1_pd(y);
        const __m256d ci = ymin + my * yscale;
        for (int x = 0; x < width; x += 4) {
            __m256d mx = _mm256_set_pd(x + 3, x + 2, x + 1, x + 0);
            __m256d cr = xmin + mx * xscale;
            __m256d zr = zero;
            __m256d zi = zero;

            int k = 0;
            // store the iterations
            __m256d mk = _mm256_set1_pd(k);

            // last Zr/Zi values -> make them accessible as float vector
            __m256d mlastZr = zero;
            __m256d mlastZi = zero;

            // distance
            __m256d dr = one;
            __m256d di = zero;

            __m256d lastDr = dr;
            __m256d lastDi = di;

            __m256d previousInsideMask = _mm256_set1_pd(0xFFFFFFFFFFFFFFFF);

            while (++k <= maxIterations) {
                /* Compute z1 from z0 */
                const __m256d zr2 = zr * zr;
                const __m256d zi2 = zi * zi;

                const __m256d insideMask = _mm256_cmp_pd(zr2+zi2, threshold, _CMP_LT_OS);

                // store last inside values of z
                // copy only if inside mask changes for the vector (xor previous and current
                const __m256d noticeZMask = _mm256_xor_pd(insideMask, previousInsideMask);
                mlastZr  = _mm256_and_pd(noticeZMask, zr) + mlastZr;
                mlastZi  = _mm256_and_pd(noticeZMask, zi) + mlastZi;
                if( calcDistance ) {
                    lastDr  = _mm256_and_pd(noticeZMask, dr) + lastDr;
                    lastDi  = _mm256_and_pd(noticeZMask, di) + lastDi;
                }
                previousInsideMask = insideMask;

                /* Early bailout? */
                if (_mm256_testz_pd(insideMask, oneminus)) {
                    break;
                }

                /* Increment k for all vectors inside */
                mk = _mm256_and_pd(insideMask, one) + mk;

                if ( calcDistance) {
                    const __m256d new_dr = 2.0 * (zr * dr - zi * di) + 1.0;
                    di = 2.0 * (zr * di + zi * dr);
                    dr = new_dr;
                }

                zi = 2*zr*zi+ci;
                zr = zr2 - zi2 + cr;
            }

            // convert counter to int and make it accessible via array index
            union {
                int i[4];
                __m128i m;
            } vCount;
            vCount.m = _mm256_cvtpd_epi32(mk);

            double tLastZrs[4];
            double tLastZis[4];

            _mm256_storeu_pd(tLastZrs, mlastZr);
            _mm256_storeu_pd(tLastZis, mlastZi);

            const int tIndex = x + y * width;
            for ( int i=0; i<4 && x+i<width; i++ ) {
                iters[tIndex+i] = vCount.i[i];
                lastZrs[tIndex+i] = tLastZrs[i];
                lastZis[tIndex+i] = tLastZis[i];
            }

            if ( calcDistance) {
                double tLastDrs[4];
                double tLastDis[4];

                _mm256_storeu_pd(tLastDrs, lastDr);
                _mm256_storeu_pd(tLastDis, lastDi);

                for ( int i=0; i<4 && x+i<width; i++ ) {
                    distancesR[tIndex+i] = tLastDrs[i];
                    distancesI[tIndex+i] = tLastDis[i];
                }
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
            double *distancesR,
            double *distancesI,
            bool calcDistance,
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
    const __m256 zero = _mm256_set1_ps(0);
    const __m256 oneminus = _mm256_set1_ps(-1);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < height; y++) {
        // as long as the assignment loop is failing, we calc some pixels less to avoid writing outside array limits
        const __m256 my = _mm256_set1_ps(y);
        const __m256 ci = ymin + my * yscale;
        for (int x = 0; x < width; x += 8) {
            __m256 mx = _mm256_set_ps(x + 7, x + 6, x + 5, x + 4, x + 3, x + 2, x + 1, x + 0);
            __m256 cr = xmin + mx * xscale;
            __m256 zr = zero;
            __m256 zi = zero;

            int k = 0;
            // store the iterations
            __m256 mk = _mm256_set1_ps(k);

            // last Zr/Zi values -> make them accessible as float vector
            __m256 mlastZr = zero;
            __m256 mlastZi = zero;

            // distance
            __m256 dr = one;
            __m256 di = zero;

            __m256 lastDr = dr;
            __m256 lastDi = di;

            __m256 previousInsideMask = _mm256_set1_ps(0xFFFFFFFFFFFFFFFF);

            while (++k <= maxIterations) {
                /* Compute z1 from z0 */
                const __m256 zr2 = zr * zr;
                const __m256 zi2 = zi * zi;

                const __m256 insideMask = _mm256_cmp_ps(zr2+zi2, threshold, _CMP_LT_OS);

                // store last inside values of z
                // copy only if inside mask changes for the vector (xor previous and current
                const __m256 noticeZMask = _mm256_xor_ps(insideMask, previousInsideMask);
                mlastZr  = _mm256_and_ps(noticeZMask, zr) + mlastZr;
                mlastZi  = _mm256_and_ps(noticeZMask, zi) + mlastZi;
                if( calcDistance ) {
                    lastDr  = _mm256_and_ps(noticeZMask, dr) + lastDr;
                    lastDi  = _mm256_and_ps(noticeZMask, di) + lastDi;
                }
                previousInsideMask = insideMask;

                /* Early bailout? */
                if (_mm256_testz_ps(insideMask, oneminus)) {
                    break;
                }

                /* Increment k for all vectors inside */
                mk = _mm256_and_ps(insideMask, one) + mk;

                if ( calcDistance) {
                    const __m256 new_dr = 2.0 * (zr * dr - zi * di) + 1.0;
                    di = 2.0 * (zr * di + zi * dr);
                    dr = new_dr;
                }

                zi = 2*zr*zi+ci;
                zr = zr2 - zi2 + cr;
            }

            // convert counter to int and make it accessible via array index
            union {
                int i[8];
                __m256i m;
            } vCount;
            vCount.m = _mm256_cvtps_epi32(mk);

            float tLastZrs[8];
            float tLastZis[8];

            _mm256_storeu_ps(tLastZrs, mlastZr);
            _mm256_storeu_ps(tLastZis, mlastZi);

            const int tIndex = x + y * width;
            for ( int i=0; i<8 && x+i<width; i++ ) {
                iters[tIndex+i] = vCount.i[i];
                lastZrs[tIndex+i] = tLastZrs[i];
                lastZis[tIndex+i] = tLastZis[i];
            }

            if ( calcDistance) {
                float tLastDrs[8];
                float tLastDis[8];

                _mm256_storeu_ps(tLastDrs, lastDr);
                _mm256_storeu_ps(tLastDis, lastDi);

                for ( int i=0; i<8 && x+i<width; i++ ) {
                    distancesR[tIndex+i] = tLastDrs[i];
                    distancesI[tIndex+i] = tLastDis[i];
                }
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
        const double ci = yStart + y*yInc;
        for (int x = 0; x < width; x ++) {
            const double cr = xStart + x*xInc;

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
