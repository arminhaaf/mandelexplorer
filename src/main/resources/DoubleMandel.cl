#pragma OPENCL EXTENSION cl_khr_fp64 : enable

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
      int calcDistance,
      double4 area,
      int maxIterations,
      double sqrEscapeRadius
      ) {

   const double x = area.x + X*area.z;
   const double y = area.y + Y*area.w;

   const double escape = sqrEscapeRadius;

   double zr = x;
   double zi = y;
   double new_zr = 0.0f;

   // distance
   double dr = 1;
   double di = 0;
   double new_dr;

   const bool tCalcDistance = calcDistance>0;

   int count = 0;
   for (; count<maxIterations; count++){
        const double zrsqr = zr * zr;
        const double zisqr = zi * zi;

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