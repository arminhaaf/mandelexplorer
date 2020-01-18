#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mandel.h"

#define SPLIT  134217729.0 // 2^27+1, for IEEE double


 DD DD_mul(const DD pFF1, const DD pFF2) {
    const double hi = pFF1.hi;
    const double lo = pFF1.lo;
    const double yhi = pFF2.hi;
    const double ylo = pFF2.lo;


    double hx, tx, hy, ty, C, c;
    C = SPLIT * hi;
    hx = C - hi;
    c = SPLIT * yhi;
    hx = C - hx;
    tx = hi - hx;
    hy = c - yhi;
    C = hi * yhi;
    hy = c - hy;
    ty = yhi - hy;
    c = ((((hx * hy - C) + hx * ty) + tx * hy) + tx * ty) + (hi * ylo + lo * yhi);
//    c = fma(tx,ty,fma(tx,hy, fma(hx,ty,fma(hx,hy, - C)))) + fma(hi,ylo,lo * yhi);
    double zhi = C + c;
    hx = C - zhi;
    double zlo = c + hx;

    return (DD){zhi,zlo};
}

 DD DD_mulDouble(const DD pFF1, const double pDouble) {
    const double hi = pFF1.hi;
    const double lo = pFF1.lo;
    const double yhi = pDouble;

    double hx, tx, hy, ty, C, c;
    C = SPLIT * hi;
    hx = C - hi;
    c = SPLIT * yhi;
    hx = C - hx;
    tx = hi - hx;
    hy = c - yhi;
    C = hi * yhi;
    hy = c - hy;
    ty = yhi - hy;
    c = ((((hx * hy - C) + hx * ty) + tx * hy) + tx * ty) + (lo * yhi);
//    c = fma(lo, yhi, fma(tx, ty, fma(tx, hy, fma(hx, ty, fma(hx, hy, -C)))));
    double zhi = C + c;
    hx = C - zhi;
    double zlo = c + hx;

    return (DD){zhi,zlo};
}


 DD DD_add(const DD pFF1, const DD pFF2) {
    const double hi = pFF1.hi;
    const double lo = pFF1.lo;
    const double yhi = pFF2.hi;
    const double ylo = pFF2.lo;

    double H, h, T, t, S, s, e, f;
    S = hi + yhi;
    T = lo + ylo;
    e = S - hi;
    f = T - lo;
    s = S - e;
    t = T - f;
    s = (yhi - e) + (hi - s);
    t = (ylo - f) + (lo - t);
    e = s + T;
    H = S + e;
    h = e + (S - H);
    e = t + h;

    const double zhi = H + e;
    const double zlo = e + (H - zhi);

    return (DD){zhi,zlo};
}

 DD DD_addDouble(const DD pFF1, const  double y) {
    double hi = pFF1.hi;
    double lo = pFF1.lo;

    double H, h, S, s, e, f;
    S = hi + y;
    e = S - hi;
    s = S - e;
    s = (y - e) + (hi - s);
    f = s + lo;
    H = S + f;
    h = f + (S - H);
    hi = H + h;
    lo = h + (H - hi);

    return (DD){hi,lo};
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
mandel_dd(unsigned int *iters,
          double *lastZrs,
          double *lastZis,
          double *distancesR,
          double *distancesI,
          bool calcDistance,
          const int width,
          const int height,
          const double xStartHi,
          const double xStartLo,
          const double yStartHi,
          const double yStartLo,
          const double xIncHi,
          const double xIncLo,
          const double yIncHi,
          const double yIncLo,
          const unsigned int maxIterations,
          const double sqrEscapeRadius)
{
    checkCompilerOptimization();

    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < height; y++) {
        // as long as the assignment loop is failing, we calc some pixels less to avoid writing outside array limits
        const DD ci = DD_add((DD) {yStartHi, yStartLo}, DD_mulDouble((DD) {yIncHi, yIncLo}, y));
        for (int x = 0; x < width; x ++) {
            const DD cr = DD_add((DD) {xStartHi, xStartLo}, DD_mulDouble((DD) {xIncHi, xIncLo}, x));

            DD zr = cr;
            DD zi = ci;

            // distance
            DD dr = (DD){1,0};
            DD di = (DD){0,0};

            const bool tCalcDistance = calcDistance>0;

            unsigned int count = 0;

            for (; count<maxIterations; count++){
                const DD zrsqr = DD_mul(zr, zr);
                const DD zisqr = DD_mul(zi, zi);

                if (DD_add(zrsqr, zisqr).hi >= (double) sqrEscapeRadius) {
                    break;
                }

                if ( tCalcDistance) {
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

            const int tIndex = x + y * width;
            iters[tIndex]  = count;
            lastZrs[tIndex] = (double)zr.hi + (double)zr.lo;
            lastZis[tIndex] = (double)zi.hi + (double)zi.lo;
            if ( tCalcDistance ) {
                distancesR[tIndex] = (double)dr.hi + (double)dr.lo;
                distancesI[tIndex] = (double)di.hi + (double)di.lo;
            }
        }
    }

}

