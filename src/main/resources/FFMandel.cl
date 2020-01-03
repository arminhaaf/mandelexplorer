#pragma OPENCL EXTENSION cl_khr_fp64 : enable

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

    float hx, tx, hy, ty, C, c;
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
    const float zhi = C + c;
    hx = C - zhi;
    const float zlo = c + hx;

    return (float2)(zhi,zlo);
}

inline float2 mulFloat(const float2 pFF1, const float pFloat) {
    const float hi = pFF1.x;
    const float lo = pFF1.y;
    const float yhi = pFloat;

    float hx, tx, hy, ty, C, c;
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
    const float zhi = C + c;
    hx = C - zhi;
    const float zlo = c + hx;

    return (float2)(zhi,zlo);
}


inline float2 add(const float2 pFF1,  const float2 pFF2) {
    const float hi = pFF1.x;
    const float lo = pFF1.y;
    const float yhi = pFF2.x;
    const float ylo = pFF2.y;

    float H, h, T, t, S, s, e, f;
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

    const float zhi = H + e;
    const float zlo = e + (H - zhi);

    return (float2)(zhi,zlo);
}

inline float2 addFloat(const float2 pFF1,const  float y) {
    float hi = pFF1.x;
    float lo = pFF1.y;

    float H, h, S, s, e, f;
    S = hi + y;
    e = S - hi;
    s = S - e;
    s = (y - e) + (hi - s);
    f = s + lo;
    H = S + f;
    h = f + (S - H);
    hi = H + h;
    lo = h + (H - hi);

    return (float2)(hi,lo);
}

inline float2 sub(const float2 pFF1,const  float2 pFF2) {
    return add(pFF1, (float2)(-pFF2.x, -pFF2.y));
}


__kernel void computeMandelBrot(
       __global int *iters,
       __global double *lastValuesR,
       __global double *lastValuesI,
       __global double *distancesR,
       __global double *distancesI,
       int calcDistance,

       __global float *xStart,
       __global float *yStart,
       __global float *xInc,
       __global float *yInc,
       int maxIterations,
       double sqrEscapeRadius
       ) {

    const float2 x = add((float2)(xStart[0], xStart[1]),mulFloat((float2)(xInc[0],xInc[1]),X));
    const float2 y = add((float2)(yStart[0], yStart[1]),mulFloat((float2)(yInc[0],yInc[1]),Y));

    const float escape = (float)sqrEscapeRadius;

    float2 zr = x;
    float2 zi = y;

    float2 tmp;

    // distance
    float2 dr = (float2)(1,0);
    float2 di = (float2)(0,0);
    float2 new_dr;

    const bool tCalcDistance = calcDistance>0;

    int count = 0;

    for (; count<maxIterations; count++){
        const float2 zrsqr = mul(zr,zr);
        const float2 zisqr = mul(zi,zi);

        if ( add(zrsqr,zisqr).x >= escape ) {
            break;
        }

        if ( tCalcDistance) {
//            new_dr = 2.0f * (zr * dr - zi * di) + 1.0f;
            new_dr = addFloat(mulFloat(sub(mul(zr,dr),mul(zi,di)),2.0f),1.0f);
//            di = 2.0f * (zr * di + zi * dr);
            di = mulFloat(add(mul(zr,di),mul(zi,dr)),2.0f);
            dr = new_dr;
        }

        tmp = add(sub(zrsqr,zisqr),x);
        zi = add(mulFloat(mul(zr,zi),2.0f),y);
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