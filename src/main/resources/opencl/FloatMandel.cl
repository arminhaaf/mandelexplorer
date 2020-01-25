#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define MODE_MANDEL 1
#define MODE_MANDEL_DISTANCE 2
#define MODE_JULIA 3

#define WIDTH get_global_size(0)
#define HEIGHT get_global_size(1)
#define X get_global_id(0)
#define Y get_global_id(1)

__kernel void compute(
      __global int *iters,
      __global double *lastValuesR,
      __global double *lastValuesI,
      __global double *distancesR,
      __global double *distancesI,
      int mode,
      double4 area,
      double2 julia,
      int maxIterations,
      double sqrEscapeRadius
      ) {

   const float x = area.x + X*area.z;
   const float y = area.y + Y*area.w;
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
   for (; count<maxIterations; count++){
        const float zrsqr = zr * zr;
        const float zisqr = zi * zi;

        if ( (zrsqr + zisqr) >= escape ) {
            break;
        }

        if ( mode == MODE_MANDEL_DISTANCE) {
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
        if ( mode == MODE_MANDEL_DISTANCE) {
      distancesR[tIndex] = (double)dr;
      distancesI[tIndex] = (double)di;
   }
}