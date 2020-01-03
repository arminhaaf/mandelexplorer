#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define WIDTH get_global_size(0)
#define HEIGHT get_global_size(1)
#define X get_global_id(0)
#define Y get_global_id(1)

__kernel void computeMandelBrot(
      __global int *iters,
      __global double *lastValuesR,
      __global double *lastValuesI,
      __global double *distancesR,
      __global double *distancesI,
      int calcDistance,

      float xStart,
      float yStart,
      float xInc,
      float yInc,
      int maxIterations,
      float sqrEscapeRadius
      ) {

   const float x = xStart + X*xInc;
   const float y = yStart + Y*yInc;

   const float escape = sqrEscapeRadius;

   float zr = x;
   float zi = y;
   float new_zr = 0.0f;

   // distance
   float dr = 1;
   float di = 0;
   float new_dr;

   const bool tCalcDistance = calcDistance>0;

   int count = 0;
   for (; count<maxIterations; count++){
        const float zrsqr = zr * zr;
        const float zisqr = zi * zi;

        if ( (zrsqr + zisqr) >= escape ) {
            break;
        }

        if ( tCalcDistance) {
            new_dr = 2.0f * (zr * dr - zi * di) + 1.0f;
            di = 2.0f * (zr * di + zi * dr);
            dr = new_dr;
        }

        new_zr = (zrsqr - zisqr) + x;
        zi = ((2.0f * zr) * zi) + y;
        zr = new_zr;

        //If in a periodic orbit, assume it is trapped
        if (zr == 0.0 && zi == 0.0) {
            count = maxIterations;
            break;
        }
   }
   const int tIndex = X + Y * WIDTH;
   iters[tIndex]  = count;
   lastValuesR[tIndex] = (double)zr;
   lastValuesI[tIndex] = (double)zi;
   if ( tCalcDistance ) {
      distancesR[tIndex] = (double)dr;
      distancesI[tIndex] = (double)di;
   }
}