#pragma once


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
                                const double sqrEscapeRadius);

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
                                const float sqrEscapeRadius);


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
                                const double sqrEscapeRadius);


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
            const double sqrEscapeRadius);

