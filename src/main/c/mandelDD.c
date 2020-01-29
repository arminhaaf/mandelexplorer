#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mandel.h"


 DD DD_mul(const DD pFF1, const DD pFF2) {
    const double hi = pFF1.hi;
    const double lo = pFF1.lo;
    const double yhi = pFF2.hi;
    const double ylo = pFF2.lo;


       double t, tau, u, v, w;

           t = hi * yhi;            /* Highest order double term.  */

           if (t == 0) {
               return (DD){0,0};
           }

           tau = fma(hi, yhi, -t);
           v = hi * ylo;
           w = lo * yhi;
           tau += v + w;        /* Add in other second-order terms.	 */
           u = t + tau;

       return (DD){u, (t - u) + tau};
}

 DD DD_mulDouble(const DD pFF1, const double pDouble) {
    const double hi = pFF1.hi;
    const double lo = pFF1.lo;
    const double yhi = pDouble;

       double t, tau, u, w;

           t = hi * yhi;            /* Highest order double term.  */

           if (t == 0) {
               return (DD){0,0};
           }

           tau = fma(hi, yhi, -t);
           w = lo * yhi;
           tau += w;        /* Add in other second-order terms.	 */
           u = t + tau;

       return (DD){u, (t - u) + tau};}


 DD DD_add(const DD pFF1, const DD pFF2) {
    const double hi = pFF1.hi;
    const double lo = pFF1.lo;
    const double yhi = pFF2.hi;
    const double ylo = pFF2.lo;

    double z, q, zz, xh;

           z = hi + yhi;

           q = hi - z;
           zz = q + yhi + (hi - (q + z)) + lo + ylo;

           /* Keep -0 result.  */
           if (zz == 0.0) {
               return (DD){z,0};
           }

           xh = z + zz;

    return (DD){xh,z - xh + zz};
}

 DD DD_addDouble(const DD pFF1, const  double y) {
    double hi = pFF1.hi;
    double lo = pFF1.lo;

    double z, q, zz, xh;

           z = hi + y;

           q = hi - z;
           zz = q + y + (hi - (q + z)) + lo;

           /* Keep -0 result.  */
           if (zz == 0.0) {
               return (DD){z,0};
           }

           xh = z + zz;

    return (DD){xh,z - xh + zz};
}

 DD DD_sub(const DD pFF1, const  DD pFF2) {
    return DD_add(pFF1, (DD) {-pFF2.hi, -pFF2.lo});
}

void checkCompilerOptimization() {
   DD y = (DD){2.9615004935834156e-03, -1.8408960875370855e-20};
   DD erg = DD_mulDouble(y, 1.0120000000000000e+03);

   if ( erg.lo!=4.2085453253312943e-17) {
        printf("compiler break DD Logik -> please do not use -ffast-math or -funsafe-math-optimizations\n");
        fflush(stdout);
    }
}

void
mandel_dd(int32_t *iters,
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
    checkCompilerOptimization();

    const DD ddXStart = (DD) {xStartHi, xStartLo};
    const DD ddYStart = (DD) {yStartHi, yStartLo};
    const DD ddXInc = (DD) {xIncHi, xIncLo};
    const DD ddYInc = (DD) {yIncHi, yIncLo};
    const DD ddJuliaCr = (DD) {juliaCrHi, juliaCrLo};
    const DD ddJuliaCi = (DD) {juliaCiHi, juliaCiLo};

    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < height; y++) {
        // as long as the assignment loop is failing, we calc some pixels less to avoid writing outside array limits
        const DD tY = DD_add(ddYStart, DD_mulDouble(ddYInc, y));
        const DD ci = mode == MODE_JULIA ? ddJuliaCi : tY;
        DD tX = ddXStart;
        for (int x = 0; x < width; x++) {
            const DD cr = mode == MODE_JULIA ? ddJuliaCr : tX;

            DD zr = tX;
            DD zi = tY;

            // distance
            DD dr = (DD){1,0};
            DD di = (DD){0,0};

            int32_t count = 0;

            for (; count<maxIterations; count++){
                const DD zrsqr = DD_mul(zr, zr);
                const DD zisqr = DD_mul(zi, zi);

                if (DD_add(zrsqr, zisqr).hi >= (double) sqrEscapeRadius) {
                    break;
                }

                if ( mode == MODE_MANDEL_DISTANCE ) {
//            new_dr = 2.0f * (zr * dr - zi * di) + 1.0f;
                    DD new_dr = DD_addDouble(DD_mulDouble(DD_sub(DD_mul(zr, dr), DD_mul(zi, di)), 2.0), 1.0);
//            di = 2.0f * (zr * di + zi * dr);
                    di = DD_mulDouble(DD_add(DD_mul(zr, di), DD_mul(zi, dr)), 2.0);
                    dr = new_dr;
                }

                DD tmp = DD_add(DD_sub(zrsqr, zisqr), cr);
                zi = DD_add(DD_mulDouble(DD_mul(zr, zi), 2.0), ci);
                zr = tmp;
            }

            const int32_t tIndex = x + y * width;
            iters[tIndex]  = count;
            lastZrs[tIndex] = (double)zr.hi + (double)zr.lo;
            lastZis[tIndex] = (double)zi.hi + (double)zi.lo;
            if ( mode == MODE_MANDEL_DISTANCE ) {
                distancesR[tIndex] = (double)dr.hi + (double)dr.lo;
                distancesI[tIndex] = (double)di.hi + (double)di.lo;
            }

            tX = DD_add(tX, ddXInc);
        }
    }

}

