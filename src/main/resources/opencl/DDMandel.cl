#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define MODE_MANDEL 1
#define MODE_MANDEL_DISTANCE 2
#define MODE_JULIA 3

#define WIDTH get_global_size(0)
#define HEIGHT get_global_size(1)
#define X get_global_id(0)
#define Y get_global_id(1)

inline double2 mul(const double2 pFF1, const double2 pFF2) {
    const double hi = pFF1.x;
    const double lo = pFF1.y;
    const double yhi = pFF2.x;
    const double ylo = pFF2.y;

    double t, tau, u, v, w;

        t = hi * yhi;            /* Highest order double term.  */

        if (t == 0) {
            return (double2)(0,0);
        }

        tau = fma(hi, yhi, -t);
        v = hi * ylo;
        w = lo * yhi;
        tau += v + w;        /* Add in other second-order terms.	 */
        u = t + tau;

    return (double2)(u, (t - u) + tau);
}

inline double2 mulDouble(const double2 pFF1, const double pDouble) {
    const double hi = pFF1.x;
    const double lo = pFF1.y;
    const double yhi = pDouble;

        double t, tau, u, w;

        t = hi * yhi;            /* Highest order double term.  */

        if (t == 0) {
            return (double2)(0,0);
        }

        tau = fma(hi, yhi, -t);
        w = lo * yhi;
        tau += w;        /* Add in other second-order terms.	 */
        u = t + tau;

    return (double2)(u, (t - u) + tau);
}


inline double2 add(const double2 pFF1,  const double2 pFF2) {
    const double hi = pFF1.x;
    const double lo = pFF1.y;
    const double yhi = pFF2.x;
    const double ylo = pFF2.y;

        double z, q, zz, xh;

        z = hi + yhi;

        q = hi - z;
        zz = q + yhi + (hi - (q + z)) + lo + ylo;

        /* Keep -0 result.  */
        if (zz == 0.0) {
            return (double2)(z,0);
        }

        xh = z + zz;

    return (double2)(xh,z - xh + zz);
}

inline double2 addDouble(const double2 pFF1,const  double y) {
    const double hi = pFF1.x;
    const double lo = pFF1.y;
    const double yhi = y;
    
    double z, q, zz, xh;

        z = hi + yhi;

        q = hi - z;
        zz = q + yhi + (hi - (q + z)) + lo;

        /* Keep -0 result.  */
        if (zz == 0.0) {
            return (double2)(z,0);
        }

        xh = z + zz;

    return (double2)(xh,z - xh + zz);
}

inline double2 sub(const double2 pFF1,const  double2 pFF2) {
    return add(pFF1, (double2)(-pFF2.x, -pFF2.y));
}


__kernel void compute(
       __global int *iters,
       __global double *lastValuesR,
       __global double *lastValuesI,
       __global double *distancesR,
       __global double *distancesI,
       const int mode,

       const double2 xStart,
       const double2 yStart,
       const double2 juliaCr,
       const double2 juliaCi,
       const double2 xInc,
       const double2 yInc,
       const int maxIterations,
       const double sqrEscapeRadius
       ) {

    const double2 x = add((double2)(xStart.x, xStart.y),mulDouble((double2)(xInc.x,xInc.y),X));
    const double2 y = add((double2)(yStart.x, yStart.y),mulDouble((double2)(yInc.x,yInc.y),Y));

    const double2 cr = mode == MODE_JULIA ? juliaCr : x;
    const double2 ci = mode == MODE_JULIA ? juliaCi : y;

    double2 zr = x;
    double2 zi = y;
    double2 tmp;

    // distance
    double2 dr = (double2)(1,0);
    double2 di = (double2)(0,0);
    double2 new_dr;

    int count = 0;

    for (; count<maxIterations; count++){
        const double2 zrsqr = mul(zr,zr);
        const double2 zisqr = mul(zi,zi);

        if ( add(zrsqr,zisqr).x >= sqrEscapeRadius ) {
            break;
        }

        if ( mode == MODE_MANDEL_DISTANCE) {
//            new_dr = 2.0f * (zr * dr - zi * di) + 1.0f;
            new_dr = addDouble(mulDouble(sub(mul(zr,dr),mul(zi,di)),2.0),1.0);
//            di = 2.0f * (zr * di + zi * dr);
            di = mulDouble(add(mul(zr,di),mul(zi,dr)),2.0);
            dr = new_dr;
        }

        tmp = add(sub(zrsqr,zisqr),cr);
        zi = add(mulDouble(mul(zr,zi),2.0),ci);
        zr = tmp;
    }
       
    const int tIndex = X + Y * WIDTH;
    iters[tIndex]  = count;
    lastValuesR[tIndex] = (double)zr.x + (double)zr.y;
    lastValuesI[tIndex] = (double)zi.x + (double)zi.y;
    if ( mode == MODE_MANDEL_DISTANCE ) {
        distancesR[tIndex] = (double)dr.x + (double)dr.y;
        distancesI[tIndex] = (double)di.x + (double)di.y;
    }
}