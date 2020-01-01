#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define WIDTH get_global_size(0)
#define HEIGHT get_global_size(1)
#define X get_global_id(0)
#define Y get_global_id(1)

__kernel void computeMandelBrot(
      __global int *iters,
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

   float zx = x;
   float zy = y;
   float zxsqr = zx * zx;
   float zysqr = zy * zy;

   float new_zx = 0.0f;
   int count = 0;
   for (; count<maxIterations && (zxsqr + zysqr)<escape; count++){
      new_zx = (zxsqr - zysqr) + x;
      zy = ((2.0f * zx) * zy) + y;
      zx = new_zx;

       //If in a periodic orbit, assume it is trapped
                  if (zx == 0.0 && zy == 0.0) {
                      count = maxIterations;
                      break;
                  } else {
          zxsqr = zx * zx;
           zysqr = zy * zy;
                  }
   }
   iters[X + Y*WIDTH]  = count;
}