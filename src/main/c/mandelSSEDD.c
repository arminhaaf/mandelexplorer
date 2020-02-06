#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "mandelSSEDD.h"

/*


es können 2 DD auf einmal bearbeitet werden, d.h. in einem Register stehen 2 hi, im anderen die 2 lo



*/

DD2 DD2_mul(const DD2 pDD1, const DD2 pDD2) {
    const __m128d hi = pDD1.hi;
    const __m128d lo = pDD1.lo;
    const __m128d yhi = pDD2.hi;
    const __m128d ylo = pDD2.lo;
    __m128d t, tau, u, v, w;

    t = hi * yhi;            /* Highest order double term.  */

    tau = _mm_fmsub_pd(hi, yhi, t);
    v = hi * ylo;
    w = lo * yhi;
    tau += v + w;        /* Add in other second-order terms.	 */
    u = t + tau;

    const __m128d rlo = (t - u)  + tau;

    return (DD2){u, rlo};
}

DD2 DD2_mul_m128d(const DD2 pDD1, const __m128d pDouble4) {
    const __m128d hi = pDD1.hi;
    const __m128d lo = pDD1.lo;
    const __m128d yhi = pDouble4;
    __m128d t, tau, u, w;

    t = hi * yhi;            /* Highest order double term.  */

    tau = _mm_fmsub_pd(hi, yhi, t);
    w = lo * yhi;
    tau += w;        /* Add in other second-order terms.	 */
    u = t + tau;

    const __m128d rlo = (t - u)  + tau;

    return (DD2){u, rlo};
}


DD2 DD2_add(const DD2 pDD1,  const DD2 pDD2) {
    const __m128d hi = pDD1.hi;
    const __m128d lo = pDD1.lo;
    const __m128d yhi = pDD2.hi;
    const __m128d ylo = pDD2.lo;

    __m128d z, q, zz, xh;

    z = hi + yhi;

    q = hi - z;
    zz = q + yhi + (hi - (q + z)) + lo + ylo;

    xh = z + zz;

    const __m128d rlo = z - xh + zz;

    return (DD2){xh, rlo};
}

DD2 DD2_add__m128d(const DD2 pDD1, const  __m128d y) {
    __m128d hi = pDD1.hi;
    __m128d lo = pDD1.lo;

    __m128d z, q, zz, xh;

    z = hi + y;

    q = hi - z;
    zz = q + y + (hi - (q + z)) + lo;

    xh = z + zz;

    const __m128d rlo = z - xh + zz;

    return (DD2){xh, rlo};
}

DD2 DD2_sub(const DD2 pDD1,const  DD2 pDD2) {
    return DD2_add(pDD1, (DD2){-pDD2.hi, -pDD2.lo});
}

void checkCompilerOptimizationDD2() {
   DD2 y = (DD2){_mm_set1_pd(2.9615004935834156e-03),_mm_set1_pd(-1.8408960875370855e-20)};
   DD2 erg = DD2_mul_m128d(y, _mm_set1_pd(1.0120000000000000e+03));

   if ( erg.lo[0]!=4.2085453253312943e-17) {
        printf("compiler break DD Logik -> please do not use -ffast-math or -funsafe-math-optimizations\n");
        fflush(stdout);
    }
}


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
            const double sqrEscapeRadius)
{
    checkCompilerOptimizationDD2();

    const __m128d mZero = _mm_set1_pd(0);
    const __m128d mOne = _mm_set1_pd(1);
    const __m128d mOneminus = _mm_set1_pd(-1);
    const __m128d threshold = _mm_set1_pd(sqrEscapeRadius);


    const DD2 xmin = (DD2){_mm_set1_pd(xStartHi), _mm_set1_pd(xStartLo)};
    const DD2 ymin = (DD2){_mm_set1_pd(yStartHi), _mm_set1_pd(yStartLo)};
    const DD2 xScale = (DD2){_mm_set1_pd(xIncHi), _mm_set1_pd(xIncLo)};
    const DD2 yScale = (DD2){_mm_set1_pd(yIncHi), _mm_set1_pd(yIncLo)};
    const DD2 juliaCr = (DD2){_mm_set1_pd(juliaCrHi), _mm_set1_pd(juliaCrLo)};
    const DD2 juliaCi = (DD2){_mm_set1_pd(juliaCiHi), _mm_set1_pd(juliaCiLo)};
    const DD2 zero = (DD2){mZero, mZero};
    const DD2 one = (DD2){mOne, mZero};
    const DD2 xInc = DD2_mul_m128d(xScale, _mm_set1_pd(2));

    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < height; y++) {
        // as long as the assignment loop is failing, we calc some pixels less to avoid writing outside array limits
        const DD2 tY = DD2_add(ymin,DD2_mul_m128d(yScale,_mm_set1_pd(y)));
        const DD2 ci = mode == MODE_JULIA ? juliaCi : tY;
        DD2 tX = DD2_add(xmin,DD2_mul_m128d(xScale,_mm_set_pd(1,0)));
        for (int x = 0; x < width; x += 2) {
            const DD2 cr = mode == MODE_JULIA ? juliaCr : tX;

            DD2 zr = tX;
            DD2 zi = tY;

            int32_t k = 0;
            // store the iterations
            __m128d mk = _mm_set1_pd(k);

            // last Zr/Zi values -> make them accessible as float vector
            __m128d mlastZr = mZero;
            __m128d mlastZi = mZero;

            // distance
            DD2 dr = one;
            DD2 di = zero;

            __m128d lastDr = dr.hi;
            __m128d lastDi = di.hi;

            __m128d previousInsideMask = _mm_set1_pd(0xFFFFFFFFFFFFFFFF);

            while (++k <= maxIterations) {

                /* Compute z1 from z0 */
                const DD2 zr2 = DD2_mul(zr,zr);
                const DD2 zi2 = DD2_mul(zi,zi);

                const DD2 zr2zi2 = DD2_add(zr2,zi2);

                const __m128d insideMask = _mm_cmp_pd(zr2zi2.hi, threshold, _CMP_LT_OS);

                // store last inside values of z
                // copy only if inside mask changes for the vector (xor previous and current
                const __m128d noticeZMask = _mm_xor_pd(insideMask, previousInsideMask);
                mlastZr  = _mm_and_pd(noticeZMask, zr.hi) + mlastZr;
                mlastZi  = _mm_and_pd(noticeZMask, zi.hi) + mlastZi;
                if( mode == MODE_MANDEL_DISTANCE ) {
                    lastDr  = _mm_and_pd(noticeZMask, dr.hi) + lastDr;
                    lastDi  = _mm_and_pd(noticeZMask, di.hi) + lastDi;
                }
                previousInsideMask = insideMask;

                /* Early bailout? */
                if (_mm_testz_pd(insideMask, mOneminus)) {
                    break;
                }

                /* Increment k for all vectors inside */
                mk = _mm_and_pd(insideMask, mOne) + mk;

                if ( mode == MODE_MANDEL_DISTANCE) {
                    const DD2 zwergDr = DD2_sub(DD2_mul(zr,dr), DD2_mul(zi,di));
                    const DD2 zwergDi = DD2_add(DD2_mul(zr,di), DD2_mul(zi,dr));
                    dr = DD2_add(DD2_add(zwergDr,zwergDr), one);
                    di = DD2_add(zwergDi,zwergDi);
                }

                const DD2 zrzi = DD2_mul(zr,zi);
                zi = DD2_add(DD2_add(zrzi,zrzi),ci);
                zr = DD2_add(DD2_sub(zr2,zi2),cr);
            }

            // convert counter to int and make it accessible via array index
            union {
                int32_t i[2];
                __m128i m;
            } vCount;
            vCount.m = _mm_cvtpd_epi32(mk);

            double tLastZrs[2];
            double tLastZis[2];

            _mm_storeu_pd(tLastZrs, mlastZr);
            _mm_storeu_pd(tLastZis, mlastZi);

            const int tIndex = x + y * width;
            for ( int i=0; i<2 && x+i<width; i++ ) {
                iters[tIndex+i] = vCount.i[i];
                lastZrs[tIndex+i] = tLastZrs[i];
                lastZis[tIndex+i] = tLastZis[i];
            }

            if ( mode == MODE_MANDEL_DISTANCE) {
                double tLastDrs[2];
                double tLastDis[2];

                _mm_storeu_pd(tLastDrs, lastDr);
                _mm_storeu_pd(tLastDis, lastDi);

                for ( int i=0; i<2 && x+i<width; i++ ) {
                    distancesR[tIndex+i] = tLastDrs[i];
                    distancesI[tIndex+i] = tLastDis[i];
                }
            }

            tX = DD2_add(tX, xInc);
        }
    }

}