#include <stdbool.h>
#include "vectorclass/vectorclass.h"

extern "C"
void
mandel_vector(unsigned int *iters,
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

    const Vec4d vEscape = sqrEscapeRadius;
    const Vec4d vXStart = xStart;
    const Vec4d vYStart = yStart;
    const Vec4d vXInc = xInc;
    const Vec4d vYInc = yInc;
    const Vec4d vOne = 1;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < height; y++) {
        // as long as the assignment loop is failing, we calc some pixels less to avoid writing outside array limits
        const Vec4d ci = vYStart + Vec4d(y) * vYInc;
        for (int x = 0; x < width; x += 4) {
            const Vec4d cr = vXStart + Vec4d(x,x+1,x+2,x+3) * vXInc;

            Vec4d zr = cr;
            Vec4d zi = ci;

            Vec4d lastZr = 0.0;
            Vec4d lastZi = 0.0;

            // distance
            Vec4d dr = 1;
            Vec4d di = 0;

            Vec4d lastDr = dr;
            Vec4d lastDi = di;

            Vec4db previousInsideMask(true);

            int count = 0;
            Vec4d vCounts(0);
            while (++count<=maxIterations){
                const Vec4d zrsqr = zr * zr;
                const Vec4d zisqr = zi * zi;

                const Vec4db insideMask = (zrsqr + zisqr) < vEscape;
                const Vec4db outsideMask = !insideMask;

                // store last inside values of z
                // copy only if inside mask changes for the vector (xor previous and current
                const Vec4db noticeMask = insideMask ^ previousInsideMask;
                lastZr = if_add (noticeMask, lastZr, zr);
                lastZi = if_add (noticeMask, lastZi, zi);
                if( calcDistance ) {
                    lastDr = if_add (noticeMask, lastDr, dr);
                    lastDi = if_add (noticeMask, lastDi, di);
                }
                previousInsideMask = insideMask;

                if ( horizontal_and(outsideMask)) {
                    break;
                }
                vCounts = if_add ( insideMask , vCounts , vOne) ;

                if ( calcDistance) {
                    Vec4d new_dr = 2.0 * (zr * dr - zi * di) + 1.0;
                    di = 2.0 * (zr * di + zi * dr);
                    dr = new_dr;
                }

                zi = 2 * zr * zi + ci;
                zr = (zrsqr - zisqr) + cr;
            }

            const int tIndex = x + y * width;
            for ( int i=0; i<4 && x+i<width; i++ ) {
                iters[tIndex+i] = vCounts[i];
                lastZrs[tIndex+i] = lastZr[i];
                lastZis[tIndex+i] = lastZi[i];
                if ( calcDistance ) {
                    distancesR[tIndex+i] = lastDr[i];
                    distancesI[tIndex+i] = lastDi[i];
                }
            }
        }

    }
}