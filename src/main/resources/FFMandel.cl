#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define WIDTH get_global_size(0)
#define HEIGHT get_global_size(1)
#define X get_global_id(0)
#define Y get_global_id(1)

#define SPLIT  4097.0f // 2^12+1, for IEEE float


inline float computeLo(double a) {
    double temp = ((1<<27)+1) * a;
    double hi = temp - (temp - a);
    double lo = a - (float)hi;
    return (float)lo;
}

inline float computeHi(double a) {
    double temp = ((1<<27)+1) * a;
    double hi = temp - (temp - a);
    return (float)hi;
}

inline float2 mul( float2 pFF1,  float2 pFF2) {
    float hi = pFF1.x;
    float lo = pFF1.y;
    float yhi = pFF2.x;
    float ylo = pFF2.y;

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
    float zhi = C + c;
    hx = C - zhi;
    float zlo = c + hx;

    return (float2)(zhi,zlo);
}

inline float2 mulFloat( float2 pFF1, float pFloat) {
    float hi = pFF1.x;
    float lo = pFF1.y;
    float yhi = pFloat;

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
    float zhi = C + c;
    hx = C - zhi;
    float zlo = c + hx;

    return (float2)(zhi,zlo);
}


inline float2 add(float2 pFF1,  float2 pFF2) {
    float hi = pFF1.x;
    float lo = pFF1.y;
    float yhi = pFF2.x;
    float ylo = pFF2.y;

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

    float zhi = H + e;
    float zlo = e + (H - zhi);

    return (float2)(zhi,zlo);
}

inline float2 addFloat(float2 pFF1,  double y) {
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

inline float2 sub( float2 pFF1,  float2 pFF2) {
    return add(pFF1, (float2)(-pFF2.x, -pFF2.y));
}

inline float2 fromDouble(double pDouble) {
    float2 tResult;
    tResult.x = computeHi(pDouble);
    tResult.y = computeLo(pDouble);
    return tResult;
}


__kernel void computeMandelBrot(
       __global int *iters,
       __global double *lastValuesR,
       __global double *lastValuesI,
       __global double *distancesR,
       __global double *distancesI,
       int calcDistance,

       double xStart,
       double yStart,
       double xInc,
       double yInc,
       int maxIterations,
       double sqrEscapeRadius
       ) {

    const float2 x = add(fromDouble(xStart),mulFloat(fromDouble(xInc),X));
    const float2 y = add(fromDouble(yStart),mulFloat(fromDouble(yInc),Y));

    const float escape = (float)sqrEscapeRadius;

    float2 zr = x;
    float2 zi = y;
    float2 zrsqr = mul(zr,zr);
    float2 zisqr = mul(zi,zi);

    float2 tmp;

    // distance
    float2 dr = (float2)(1,0);
    float2 di = (float2)(0,0);
    float2 new_dr;

    const bool tCalcDistance = calcDistance>0;

    int count = 0;

   for (; count<maxIterations && add(zrsqr,zisqr).x < escape; count++){
      if ( tCalcDistance) {
//         new_dr = 2.0f * (zr * dr - zi * di) + 1.0f;
         new_dr = addFloat(mulFloat(sub(mul(zr,dr),mul(zi,di)),2.0f),1.0f);
//         di = 2.0f * (zr * di + zi * dr);
         di = mulFloat(add(mul(zr,di),mul(zi,dr)),2.0f);
         dr = new_dr;
      }

      tmp = add(sub(zrsqr,zisqr),x);
      zi = add(mulFloat(mul(zr,zi),2.0f),y);
      zr = tmp;

      zrsqr = mul(zr,zr);
      zisqr = mul(zi,zi);
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