#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define MODE_MANDEL 1
#define MODE_MANDEL_DISTANCE 2
#define MODE_JULIA 3

#define WIDTH get_global_size(0)
#define HEIGHT get_global_size(1)
#define X get_global_id(0)
#define Y get_global_id(1)

#define SPLIT  4097.0f // 2^12+1, for IEEE float

inline float2 mul(const float2 pFF1, const float2 pFF2) {
    const float hi = pFF1.x;
    const float lo = pFF1.y;
    const float yhi = pFF2.x;
    const float ylo = pFF2.y;

       float t, tau, u, v, w;

           t = hi * yhi;            /* Highest order double term.  */

           if (t == 0) {
               return (float2)(0,0);
           }

           tau = fma(hi, yhi, -t);
           v = hi * ylo;
           w = lo * yhi;
           tau += v + w;        /* Add in other second-order terms.	 */
           u = t + tau;

       return (float2)(u, (t - u) + tau);
}

inline float2 mulFloat(const float2 pFF1, const float pFloat) {
    const float hi = pFF1.x;
    const float lo = pFF1.y;
    const float yhi = pFloat;

        float t, tau, u, w;

        t = hi * yhi;            /* Highest order float term.  */

        if (t == 0) {
            return (float2)(0,0);
        }

        tau = fma(hi, yhi, -t);
        w = lo * yhi;
        tau += w;        /* Add in other second-order terms.	 */
        u = t + tau;

    return (float2)(u, (t - u) + tau);
}


inline float2 add(const float2 pFF1,  const float2 pFF2) {
    const float hi = pFF1.x;
    const float lo = pFF1.y;
    const float yhi = pFF2.x;
    const float ylo = pFF2.y;

        float z, q, zz, xh;

        z = hi + yhi;

        q = hi - z;
        zz = q + yhi + (hi - (q + z)) + lo + ylo;

        /* Keep -0 result.  */
        if (zz == 0.0) {
            return (float2)(z,0);
        }

        xh = z + zz;

    return (float2)(xh,z - xh + zz);
}

inline float2 addFloat(const float2 pFF1,const  float y) {
    float hi = pFF1.x;
    float lo = pFF1.y;

    float z, q, zz, xh;

        z = hi + y;

        q = hi - z;
        zz = q + y + (hi - (q + z)) + lo;

        /* Keep -0 result.  */
        if (zz == 0.0) {
            return (float2)(z,0);
        }

        xh = z + zz;

    return (float2)(xh,z - xh + zz);
}

inline float2 sub(const float2 pFF1,const  float2 pFF2) {
    return add(pFF1, (float2)(-pFF2.x, -pFF2.y));
}


__kernel void compute(
       __global int *iters,
       __global double *lastValuesR,
       __global double *lastValuesI,
       __global double *distancesR,
       __global double *distancesI,
       const int mode,

       const float2 xStart,
       const float2 yStart,
       const float2 juliaCr,
       const float2 juliaCi,
       const float2 xInc,
       const float2 yInc,
       const int maxIterations,
       const double sqrEscapeRadius
       ) {

    const float2 x = add((float2)(xStart.x, xStart.y),mulFloat((float2)(xInc.x,xInc.y),X));
    const float2 y = add((float2)(yStart.x, yStart.y),mulFloat((float2)(yInc.x,yInc.y),Y));

    const float2 cr = mode == MODE_JULIA ? juliaCr : x;
    const float2 ci = mode == MODE_JULIA ? juliaCi : y;

    const float escape = (float)sqrEscapeRadius;

    float2 zr = x;
    float2 zi = y;

    float2 tmp;

    // distance
    float2 dr = (float2)(1,0);
    float2 di = (float2)(0,0);
    float2 new_dr;

    int count = 0;

    for (; count<maxIterations; count++){
        const float2 zrsqr = mul(zr,zr);
        const float2 zisqr = mul(zi,zi);

        if ( add(zrsqr,zisqr).x >= escape ) {
            break;
        }

        if ( mode == MODE_MANDEL_DISTANCE) {
//            new_dr = 2.0f * (zr * dr - zi * di) + 1.0f;
            new_dr = addFloat(mulFloat(sub(mul(zr,dr),mul(zi,di)),2.0f),1.0f);
//            di = 2.0f * (zr * di + zi * dr);
            di = mulFloat(add(mul(zr,di),mul(zi,dr)),2.0f);
            dr = new_dr;
        }

        tmp = add(sub(zrsqr,zisqr),cr);
        zi = add(mulFloat(mul(zr,zi),2.0f),ci);
        zr = tmp;

    }
    const int tIndex = X + Y * WIDTH;
    iters[tIndex]  = count;
    lastValuesR[tIndex] = (double)zr.x + (double)zr.y;
    lastValuesI[tIndex] = (double)zi.x + (double)zi.y;
    if ( mode == MODE_MANDEL_DISTANCE) {
        distancesR[tIndex] = (double)dr.x + (double)dr.y;
        distancesI[tIndex] = (double)di.x + (double)di.y;
    }
}