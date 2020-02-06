#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "mandel.h"

// copied from https://github.com/skeeto/mandel-simd/blob/master/mandel_avx.c
// changed a lot to make it easier readable and support distance and last z


void
mandel_ssed(int32_t *iters,
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
    const __m128d threshold = _mm_set1_pd(sqrEscapeRadius);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < height; y++) {
        // as long as the assignment loop is failing, we calc some pixels less to avoid writing outside array limits
        const __m128d my = _mm_set1_pd(y);
        const __m128d tY = _mm_set1_pd(yStart) + my * _mm_set1_pd(yInc);
        const __m128d ci = mode == MODE_JULIA ? _mm_set1_pd(juliaCi) : tY;
        for (int x = 0; x < width; x += 2) {
            __m128d mx = _mm_set_pd(x + 1, x + 0);
            const __m128d tX = _mm_set1_pd(xStart) + mx * _mm_set1_pd(xInc);
            const __m128d cr = mode == MODE_JULIA ? _mm_set1_pd(juliaCr) : tX;
            __m128d zr = tX;
            __m128d zi = tY;

            int32_t k = 0;
            // store the iterations
            __m128d mk = _mm_set1_pd(k);

            // last Zr/Zi values -> make them accessible as float vector
            __m128d mlastZr = _mm_setzero_pd();
            __m128d mlastZi = _mm_setzero_pd();

            // distance
            __m128d dr = _mm_set1_pd(1);
            __m128d di = _mm_setzero_pd();

            __m128d lastDr = dr;
            __m128d lastDi = di;

            __m128d previousInsideMask = _mm_set1_pd(0xFFFFFFFFFFFFFFFF);

            while (++k <= maxIterations) {
                /* Compute z1 from z0 */
                const __m128d zr2 = zr * zr;
                const __m128d zi2 = zi * zi;

                const __m128d insideMask = _mm_cmp_pd(zr2+zi2, threshold, _CMP_LT_OS);

                // store last inside values of z
                // copy only if inside mask changes for the vector (xor previous and current
                const __m128d noticeZMask = _mm_xor_pd(insideMask, previousInsideMask);
                mlastZr  = _mm_and_pd(noticeZMask, zr) + mlastZr;
                mlastZi  = _mm_and_pd(noticeZMask, zi) + mlastZi;
                if( mode==MODE_MANDEL_DISTANCE ) {
                    lastDr  = _mm_and_pd(noticeZMask, dr) + lastDr;
                    lastDi  = _mm_and_pd(noticeZMask, di) + lastDi;
                }
                previousInsideMask = insideMask;

                /* Early bailout? */
                if (_mm_testz_pd(insideMask, _mm_set1_pd(-1))) {
                    break;
                }

                /* Increment k for all vectors inside */
                mk = _mm_and_pd(insideMask, _mm_set1_pd(1)) + mk;

                if( mode==MODE_MANDEL_DISTANCE ) {
                    const __m128d new_dr = 2.0 * (zr * dr - zi * di) + 1.0;
                    di = 2.0 * (zr * di + zi * dr);
                    dr = new_dr;
                }

                zi = 2*zr*zi+ci;
                zr = zr2 - zi2 + cr;
            }

            // convert counter to int32_t and make it accessible via array index
            union {
                int32_t i[2];
                __m128i m;
            } vCount;
            vCount.m = _mm_cvtpd_epi32(mk);

            double tLastZrs[2];
            double tLastZis[2];

            _mm_storeu_pd(tLastZrs, mlastZr);
            _mm_storeu_pd(tLastZis, mlastZi);

            const int32_t tIndex = x + y * width;
            for ( int32_t i=0; i<2 && x+i<width; i++ ) {
                iters[tIndex+i] = vCount.i[i];
                lastZrs[tIndex+i] = tLastZrs[i];
                lastZis[tIndex+i] = tLastZis[i];
            }

            if( mode==MODE_MANDEL_DISTANCE ) {
                double tLastDrs[2];
                double tLastDis[2];

                _mm_storeu_pd(tLastDrs, lastDr);
                _mm_storeu_pd(tLastDis, lastDi);

                for ( int32_t i=0; i<2 && x+i<width; i++ ) {
                    distancesR[tIndex+i] = tLastDrs[i];
                    distancesI[tIndex+i] = tLastDis[i];
                }
            }

            // next step


            //            const int32_t *counts = (int *)&mCount;
            //            const double *lastZr = (double *)&mlastZr;
            //            const double *lastZi = (double *)&mlastZi;
            // unclear why this did not work with loop
            //            for ( int32_t i=0; i<2 && x+i<width; i++ ) {
            //                iters[tIndex + i]  = counts[i];
            //                lastZrs[tIndex+i] = (double)lastZr[i];
            //                lastZis[tIndex+i] = (double)lastZi[i];
            //            }
        }
    }

}

