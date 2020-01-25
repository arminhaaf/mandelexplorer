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
      int mode,
      int4 tile,
      double4 area,
      double2 julia,
      int maxIterations,
      double sqrEscapeRadius
      ) {

   if ( X>=tile.z || Y>=tile.w) {      // tile.z is width of tile, tile.w is height of tile
        return;
   }

   const double x = area.x + X*area.z;
   const double y = area.y + Y*area.w;
   const double cr = mode == MODE_JULIA ? julia.x : x;
   const double ci = mode == MODE_JULIA ? julia.y : y;

//    if ( X % 100 == 0 )
//   printf("compute %d %d ",X,Y);
//

   const double escape = sqrEscapeRadius;

   double zr = x;
   double zi = y;
   double new_zr = 0.0f;

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

        if ( mode == MODE_MANDEL_DISTANCE) {
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
   iters[tIndex]  = count;
   lastValuesR[tIndex] = (double)zr;
   lastValuesI[tIndex] = (double)zi;
   if ( mode == MODE_MANDEL_DISTANCE ) {
      distancesR[tIndex] = (double)dr;
      distancesI[tIndex] = (double)di;
   }
}

