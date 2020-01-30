#include <stdio.h>

#define MODE_MANDEL 1
#define MODE_MANDEL_DISTANCE 2
#define MODE_JULIA 3

#define WIDTH gridDim.x*blockDim.x
#define HEIGHT gridDim.y*blockDim.y
#define X ((blockIdx.x * blockDim.x) + threadIdx.x)
#define Y ((blockIdx.y * blockDim.y) + threadIdx.y)

extern "C"
__global__ void compute(
        int *iters,
        double *lastValuesR,
        double *lastValuesI,
        double *distancesR,
        double *distancesI,
        const int mode,
        const int4 tile,
        const double4 area,
        const double2 julia,
        const int maxIterations,
        const double sqrEscapeRadius
) {

    if (X >= tile.z || Y >= tile.w) {      // tile.z is width of tile, tile.w is height of tile
        return;
    }

    const float x = (float)area.x + X * (float)area.z;
    const float y = (float)area.y + Y * (float)area.w;
    const float cr = mode == MODE_JULIA ? julia.x : x;
    const float ci = mode == MODE_JULIA ? julia.y : y;

    const float escape = sqrEscapeRadius;

    float zr = x;
    float zi = y;
    float new_zr = 0.0f;

    // distance
    float dr = 1;
    float di = 0;
    float new_dr;

    int count = 0;
    for (; count < maxIterations; count++) {
        const float zrsqr = zr * zr;
        const float zisqr = zi * zi;

        if ((zrsqr + zisqr) >= escape) {
            break;
        }

        if (mode == MODE_MANDEL_DISTANCE) {
            new_dr = 2.0f * (zr * dr - zi * di) + 1.0f;
            di = 2.0f * (zr * di + zi * dr);
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
    const int tIndex = X + Y * tile.z;  // tile.z is width of tile
    iters[tIndex] = count;
    lastValuesR[tIndex] = (double) zr;
    lastValuesI[tIndex] = (double) zi;
    if (mode == MODE_MANDEL_DISTANCE) {
        distancesR[tIndex] = (double) dr;
        distancesI[tIndex] = (double) di;
    }
}

