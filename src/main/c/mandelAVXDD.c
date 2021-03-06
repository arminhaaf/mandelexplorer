#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "mandelAVXDD.h"

/*


es können 4 DD auf einmal bearbeitet werden, d.h. in einem Register stehen 4 hi, im anderen die 4 lo



*/

DD4 DD4_mul(const DD4 pDD1, const DD4 pDD2) {
    const __m256d hi = pDD1.hi;
    const __m256d lo = pDD1.lo;
    const __m256d yhi = pDD2.hi;
    const __m256d ylo = pDD2.lo;
    __m256d t, tau, u, v, w;

    t = hi * yhi;            /* Highest order double term.  */

    tau = _mm256_fmsub_pd(hi, yhi, t);
    v = hi * ylo;
    w = lo * yhi;
    tau += v + w;        /* Add in other second-order terms.	 */
    u = t + tau;

    const __m256d rlo = (t - u)  + tau;

    return (DD4){u, rlo};
}

DD4 DD4_mul_m256d(const DD4 pDD1, const __m256d pDouble4) {
    const __m256d hi = pDD1.hi;
    const __m256d lo = pDD1.lo;
    const __m256d yhi = pDouble4;
    __m256d t, tau, u, w;

    t = hi * yhi;            /* Highest order double term.  */

    tau = _mm256_fmsub_pd(hi, yhi, t);
    w = lo * yhi;
    tau += w;        /* Add in other second-order terms.	 */
    u = t + tau;

    const __m256d rlo = (t - u)  + tau;

    return (DD4){u, rlo};
}


DD4 DD4_add(const DD4 pDD1,  const DD4 pDD2) {
    const __m256d hi = pDD1.hi;
    const __m256d lo = pDD1.lo;
    const __m256d yhi = pDD2.hi;
    const __m256d ylo = pDD2.lo;

    __m256d z, q, zz, xh;

    z = hi + yhi;

    q = hi - z;
    zz = q + yhi + (hi - (q + z)) + lo + ylo;

    xh = z + zz;

    const __m256d rlo = z - xh + zz;

    return (DD4){xh, rlo};
}

DD4 DD4_add__m256d(const DD4 pDD1, const  __m256d y) {
    __m256d hi = pDD1.hi;
    __m256d lo = pDD1.lo;

    __m256d z, q, zz, xh;

    z = hi + y;

    q = hi - z;
    zz = q + y + (hi - (q + z)) + lo;

    xh = z + zz;

    const __m256d rlo = z - xh + zz;

    return (DD4){xh, rlo};
}

DD4 DD4_sub(const DD4 pDD1,const  DD4 pDD2) {
    return DD4_add(pDD1, (DD4){-pDD2.hi, -pDD2.lo});
}

void checkCompilerOptimizationDD4() {
   DD4 y = (DD4){_mm256_set1_pd(2.9615004935834156e-03),_mm256_set1_pd(-1.8408960875370855e-20)};
   DD4 erg = DD4_mul_m256d(y, _mm256_set1_pd(1.0120000000000000e+03));

   if ( erg.lo[0]!=4.2085453253312943e-17) {
        printf("compiler break DD Logik -> please do not use -ffast-math or -funsafe-math-optimizations\n");
        fflush(stdout);
    }
}


void mandel_avxdd(
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
    checkCompilerOptimizationDD4();

    const __m256d mZero = _mm256_set1_pd(0);
    const __m256d mOne = _mm256_set1_pd(1);
    const __m256d mOneminus = _mm256_set1_pd(-1);
    const __m256d threshold = _mm256_set1_pd(sqrEscapeRadius);


    const DD4 xmin = (DD4){_mm256_set1_pd(xStartHi), _mm256_set1_pd(xStartLo)};
    const DD4 ymin = (DD4){_mm256_set1_pd(yStartHi), _mm256_set1_pd(yStartLo)};
    const DD4 xScale = (DD4){_mm256_set1_pd(xIncHi), _mm256_set1_pd(xIncLo)};
    const DD4 yScale = (DD4){_mm256_set1_pd(yIncHi), _mm256_set1_pd(yIncLo)};
    const DD4 juliaCr = (DD4){_mm256_set1_pd(juliaCrHi), _mm256_set1_pd(juliaCrLo)};
    const DD4 juliaCi = (DD4){_mm256_set1_pd(juliaCiHi), _mm256_set1_pd(juliaCiLo)};
    const DD4 zero = (DD4){mZero, mZero};
    const DD4 one = (DD4){mOne, mZero};
    const DD4 xInc = DD4_mul_m256d(xScale, _mm256_set1_pd(4));

    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < height; y++) {
        // as long as the assignment loop is failing, we calc some pixels less to avoid writing outside array limits
        const DD4 tY = DD4_add(ymin,DD4_mul_m256d(yScale,_mm256_set1_pd(y)));
        const DD4 ci = mode == MODE_JULIA ? juliaCi : tY;
        DD4 tX = DD4_add(xmin,DD4_mul_m256d(xScale,_mm256_set_pd(3,2,1,0)));
        for (int x = 0; x < width; x += 4) {
            const DD4 cr = mode == MODE_JULIA ? juliaCr : tX;

            DD4 zr = tX;
            DD4 zi = tY;

            int32_t k = 0;
            // store the iterations
            __m256d mk = _mm256_set1_pd(k);

            // last Zr/Zi values -> make them accessible as float vector
            __m256d mlastZr = mZero;
            __m256d mlastZi = mZero;

            // distance
            DD4 dr = one;
            DD4 di = zero;

            __m256d lastDr = dr.hi;
            __m256d lastDi = di.hi;

            __m256d previousInsideMask = _mm256_set1_pd(0xFFFFFFFFFFFFFFFF);

            while (++k <= maxIterations) {

                /* Compute z1 from z0 */
                const DD4 zr2 = DD4_mul(zr,zr);
                const DD4 zi2 = DD4_mul(zi,zi);

                const DD4 zr2zi2 = DD4_add(zr2,zi2);

                const __m256d insideMask = _mm256_cmp_pd(zr2zi2.hi, threshold, _CMP_LT_OS);

                // store last inside values of z
                // copy only if inside mask changes for the vector (xor previous and current
                const __m256d noticeZMask = _mm256_xor_pd(insideMask, previousInsideMask);
                mlastZr  = _mm256_and_pd(noticeZMask, zr.hi) + mlastZr;
                mlastZi  = _mm256_and_pd(noticeZMask, zi.hi) + mlastZi;
                if( mode == MODE_MANDEL_DISTANCE ) {
                    lastDr  = _mm256_and_pd(noticeZMask, dr.hi) + lastDr;
                    lastDi  = _mm256_and_pd(noticeZMask, di.hi) + lastDi;
                }
                previousInsideMask = insideMask;

                /* Early bailout? */
                if (_mm256_testz_pd(insideMask, mOneminus)) {
                    break;
                }

                /* Increment k for all vectors inside */
                mk = _mm256_and_pd(insideMask, mOne) + mk;

                if ( mode == MODE_MANDEL_DISTANCE) {
                    const DD4 zwergDr = DD4_sub(DD4_mul(zr,dr), DD4_mul(zi,di));
                    const DD4 zwergDi = DD4_add(DD4_mul(zr,di), DD4_mul(zi,dr));
                    dr = DD4_add(DD4_add(zwergDr,zwergDr), one);
                    di = DD4_add(zwergDi,zwergDi);
                }

                const DD4 zrzi = DD4_mul(zr,zi);
                zi = DD4_add(DD4_add(zrzi,zrzi),ci);
                zr = DD4_add(DD4_sub(zr2,zi2),cr);
            }

            // convert counter to int and make it accessible via array index
            union {
                int32_t i[4];
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

            if ( mode == MODE_MANDEL_DISTANCE) {
                double tLastDrs[4];
                double tLastDis[4];

                _mm256_storeu_pd(tLastDrs, lastDr);
                _mm256_storeu_pd(tLastDis, lastDi);

                for ( int i=0; i<4 && x+i<width; i++ ) {
                    distancesR[tIndex+i] = tLastDrs[i];
                    distancesI[tIndex+i] = tLastDis[i];
                }
            }

            tX = DD4_add(tX, xInc);
        }
    }

}