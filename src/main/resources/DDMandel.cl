#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define WIDTH get_global_size(0)
#define HEIGHT get_global_size(1)
#define X get_global_id(0)
#define Y get_global_id(1)

inline double2 mul(const double2 pFF1, const double2 pFF2) {
    double hi = pFF1.x;
    double lo = pFF1.y;
    const double yhi = pFF2.x;
    const double ylo = pFF2.y;

    const double H = hi * yhi;
    double L = hi * yhi - H;  //  FMA
    L = lo * yhi + L;  //  FMA
    lo = hi * ylo + L;  //  FMA
    hi = H;

    return (double2)(hi,lo);
}

inline double2 mulDouble(const double2 pFF1, const double pDouble) {
    double hi = pFF1.x;
    double lo = pFF1.y;
    const double yhi = pDouble;

    const double H = hi * yhi;
    double L = hi * yhi - H;  //  FMA
    L = lo * yhi + L;  //  FMA
    lo = L;  //  FMA
    hi = H;

    return (double2)(hi,lo);
}


inline double2 add(const double2 pFF1,  const double2 pFF2) {
    const double hi = pFF1.x;
    const double lo = pFF1.y;
    const double yhi = pFF2.x;
    const double ylo = pFF2.y;

    double IH = fabs(fmax(hi, yhi));   //  AVX512-DQ "vrangepd"
    double IL = fabs(fmin(hi, yhi));   //  AVX512-DQ "vrangepd"
    double H = hi + yhi;
    double L = H - IH;
    L = IL - L;
    L = L + lo;

     return (double2)(H, L+ylo);
}

inline double2 addDouble(const double2 pFF1,const  double pHi) {
    double hi = pFF1.x;
    double lo = pFF1.y;

    double IH = fabs(fmax(hi, pHi));   //  AVX512-DQ "vrangepd"
    double IL = fabs(fmin(hi, pHi));   //  AVX512-DQ "vrangepd"
    double H = hi + pHi;
    double L = H - IH;
    L = IL - L;
    L = L + lo;
    return (double2)(H,L);
}

inline double2 sub(const double2 pFF1,const  double2 pFF2) {
    return add(pFF1, (double2)(-pFF2.x, -pFF2.y));
}


__kernel void computeMandelBrot(
       __global int *iters,
       __global double *lastValuesR,
       __global double *lastValuesI,
       __global double *distancesR,
       __global double *distancesI,
       int calcDistance,

       __global double *xStart,
       __global double *yStart,
       __global double *xInc,
       __global double *yInc,
       int maxIterations,
       double sqrEscapeRadius
       ) {

    const double2 x = add((double2)(xStart[0], xStart[1]),mulDouble((double2)(xInc[0],xInc[1]),X));
    const double2 y = add((double2)(yStart[0], yStart[1]),mulDouble((double2)(yInc[0],yInc[1]),Y));

    const double escape = (double)sqrEscapeRadius;

    double2 zr = x;
    double2 zi = y;
    double2 tmp;

    // distance
    double2 dr = (double2)(1,0);
    double2 di = (double2)(0,0);
    double2 new_dr;

    const bool tCalcDistance = calcDistance>0;

    int count = 0;

    for (; count<maxIterations; count++){
        const double2 zrsqr = mul(zr,zr);
        const double2 zisqr = mul(zi,zi);

        if ( add(zrsqr,zisqr).x >= escape ) {
            break;
        }

        if ( tCalcDistance) {
//            new_dr = 2.0f * (zr * dr - zi * di) + 1.0f;
            new_dr = addDouble(mulDouble(sub(mul(zr,dr),mul(zi,di)),2.0),1.0);
//            di = 2.0f * (zr * di + zi * dr);
            di = mulDouble(add(mul(zr,di),mul(zi,dr)),2.0);
            dr = new_dr;
        }

        tmp = add(sub(zrsqr,zisqr),x);
        zi = add(mulDouble(mul(zr,zi),2.0),y);
        zr = tmp;
    }
       
    const int tIndex = X + Y * WIDTH;
    iters[tIndex]  = count;
    lastValuesR[tIndex] = (double)zr.x + (double)zr.y;
    lastValuesI[tIndex] = (double)zi.x + (double)zi.y;
    if ( tCalcDistance ) {
        distancesR[tIndex] = (double)dr.x + (double)dr.y;
        distancesI[tIndex] = (double)di.x + (double)di.y;
    }
}