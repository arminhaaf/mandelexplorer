#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define WIDTH get_global_size(0)
#define HEIGHT get_global_size(1)
#define X get_global_id(0)
#define Y get_global_id(1)

#define SPLIT  134217729.0 // 2^27+1, for IEEE double

inline double2 mul(const double2 pFF1, const double2 pFF2) {
    const double hi = pFF1.x;
    const double lo = pFF1.y;
    const double yhi = pFF2.x;
    const double ylo = pFF2.y;

    double hx, tx, hy, ty, C, c;
    C = SPLIT * hi;
    hx = C - hi;
    c = SPLIT * yhi;
    hx = C - hx;
    tx = hi - hx;
    hy = c - yhi;
    C = hi * yhi;
    hy = c - hy;
    ty = yhi - hy;
    c = ((((hx * hy - C) + hx * ty) + tx * hy) + tx * ty) + (hi * ylo + lo * yhi);

    const double zhi = C + c;
    hx = C - zhi;
    const double zlo = c + hx;

    return (double2)(zhi,zlo);
}

inline double2 mulDouble(const double2 pFF1, const double pDouble) {
    const double hi = pFF1.x;
    const double lo = pFF1.y;
    const double yhi = pDouble;

    double hx, tx, hy, ty, C, c;
    C = SPLIT * hi;
    hx = C - hi;
    c = SPLIT * yhi;
    hx = C - hx;
    tx = hi - hx;
    hy = c - yhi;
    C = hi * yhi;
    hy = c - hy;
    ty = yhi - hy;
    c = ((((hx * hy - C) + hx * ty) + tx * hy) + tx * ty) + (lo * yhi);

    const double zhi = C + c;
    hx = C - zhi;
    const double zlo = c + hx;

    return (double2)(zhi,zlo);
}


inline double2 add(const double2 pFF1,  const double2 pFF2) {
    const double hi = pFF1.x;
    const double lo = pFF1.y;
    const double yhi = pFF2.x;
    const double ylo = pFF2.y;

    double H, h, T, t, S, s, e, f;
    S = hi + yhi;
    T = lo + ylo;
    e = S - hi;
    f = T - lo;
    s = S - e;
    t = T - f;
    s = (yhi - e) + (hi - s);
    t = (ylo - f) + (lo - t);
    e = s + T;
    H = S + e;
    h = e + (S - H);
    e = t + h;

    const double zhi = H + e;
    const double zlo = e + (H - zhi);

    return (double2)(zhi,zlo);
}

inline double2 addDouble(const double2 pFF1,const  double y) {
    double hi = pFF1.x;
    double lo = pFF1.y;

    double H, h, S, s, e, f;
    S = hi + y;
    e = S - hi;
    s = S - e;
    s = (y - e) + (hi - s);
    f = s + lo;
    H = S + f;
    h = f + (S - H);
    hi = H + h;
    lo = h + (H - hi);

    return (double2)(hi,lo);
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