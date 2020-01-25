#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define WIDTH get_global_size(0)
#define HEIGHT get_global_size(1)
#define X get_global_id(0)
#define Y get_global_id(1)

inline float4 qfAssign(float value)
{
    return (float4)(value, 0.0f, 0.0f, 0.0f);
}

inline float computeLo(double a)
{
    double temp = ((1 << 27) + 1) * a;
    double hi = temp - (temp - a);
    double lo = a - (float)hi;
    return (float)lo;
}

inline float computeHi(double a)
{
    double temp = ((1 << 27) + 1) * a;
    double hi = temp - (temp - a);
    return (float)hi;
}

inline float4 fromDouble(double pDouble)
{
    return (float4)(computeHi(pDouble), computeLo(pDouble), 0.0f, 0.0f);
}

inline float4 qfAssign2(float2 value)
{
    return (float4)(value.x, value.y, 0.0f, 0.0f);
}

inline float4 qfNegate(float4 value)
{
    return (float4)(-value.x, -value.y, -value.z, -value.w);
}

inline float two_sum(float a, float b, float *err)
{
    float s = a + b;
    float bb = s - a;
    *err = (a - (s - bb)) + (b - bb);

    return s;
}

inline void three_sum(float *a, float *b, float *c)
{
    float t1, t2, t3;
    t1 = two_sum(*a, *b, &t2);
    *a = two_sum(*c, t1, &t3);
    *b = two_sum(t2, t3, c);
}

inline void three_sum2(float *a, float *b, float *c)
{
    float t1, t2, t3;
    t1 = two_sum(*a, *b, &t2);
    *a = two_sum(*c, t1, &t3);
    *b = t2 + t3;
}

inline float quick_two_sum(float a, float b, float *err)
{
    float s = a + b;
    *err = b - (s - a);
    return s;
}

inline void renorm(float *c0, float *c1,
                   float *c2, float *c3, float *c4)
{
    float s0, s1, s2 = 0.0f, s3 = 0.0f;
    s0 = quick_two_sum(*c3, *c4, c4);
    s0 = quick_two_sum(*c2, s0, c3);
    s0 = quick_two_sum(*c1, s0, c2);
    *c0 = quick_two_sum(*c0, s0, c1);
    s0 = *c0;
    s1 = *c1;

    if (s1 != 0.0f)
    {
        s1 = quick_two_sum(s1, *c2, &s2);
        if (s2 != 0.0f)
        {
            s2 = quick_two_sum(s2, *c3, &s3);
            if (s3 != 0.0f)
            {
                s3 += *c4;
            }
            else
            {
                s2 = quick_two_sum(s2, *c4, &s3);
            }
        }
        else
        {
            s1 = quick_two_sum(s1, *c3, &s2);
            if (s2 != 0.0f)
            {
                s2 = quick_two_sum(s2, *c4, &s3);
            }
            else
            {
                s1 = quick_two_sum(s1, *c4, &s2);
            }
        }
    }
    else
    {
        s0 = quick_two_sum(s0, *c2, &s1);
        if (s1 != 0.0f)
        {
            s1 = quick_two_sum(s1, *c3, &s2);
            if (s2 != 0.0f)
            {
                s2 = quick_two_sum(s2, *c4, &s3);
            }
            else
            {
                s1 = quick_two_sum(s1, *c4, &s2);
            }
        }
        else
        {
            s0 = quick_two_sum(s0, *c3, &s1);
            if (s1 != 0.0f)
            {
                s1 = quick_two_sum(s1, *c4, &s2);
            }
            else
            {
                s0 = quick_two_sum(s0, *c4, &s1);
            }
        }
    }

    *c0 = s0;
    *c1 = s1;
    *c2 = s2;
    *c3 = s3;
}

inline void qfAdd(float4 *sum, const float4 a, const float4 b)
{
    float s0, s1, s2, s3;
    float t0, t1, t2, t3;

  float v0, v1, v2, v3;
  float u0, u1, u2, u3;
  float w0, w1, w2, w3;

  s0 = a.x + b.x;
  s1 = a.y + b.y;
  s2 = a.z + b.z;
  s3 = a.w + b.w;

  v0 = s0 - a.x;
  v1 = s1 - a.y;
  v2 = s2 - a.z;
  v3 = s3 - a.w;

  u0 = s0 - v0;
  u1 = s1 - v1;
  u2 = s2 - v2;
  u3 = s3 - v3;

  w0 = a.x - u0;
  w1 = a.y - u1;
  w2 = a.z - u2;
  w3 = a.w - u3;

  u0 = b.x - v0;
  u1 = b.y - v1;
  u2 = b.z - v2;
  u3 = b.w - v3;

  t0 = w0 + u0;
  t1 = w1 + u1;
  t2 = w2 + u2;
  t3 = w3 + u3;

    s1 = two_sum(s1, t0, &t0);
    three_sum(&s2, &t0, &t1);
    three_sum2(&s3, &t0, &t2);
    t0 = t0 + t1 + t3;
    renorm(&s0, &s1, &s2, &s3, &t0);
    (*sum).x = s0;
    (*sum).y = s1;
    (*sum).z = s2;
    (*sum).w = s3;
}

inline void split(float a, float *hi, float *lo)
{

    float temp = ((1<<12)+1) * a;

    *hi = temp - (temp - a);
    *lo = a - *hi;
}

inline float two_prod(float a, float b, float *err)
{
    float a_hi, a_lo, b_hi, b_lo;
    float p = a * b;
    split(a, &a_hi, &a_lo);
    split(b, &b_hi, &b_lo);
    *err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;

    return p;
}

inline void qfMul(float4 *prod, const float4 a, const float4 b)
{
    float p0, p1, p2, p3, p4, p5;
    float q0, q1, q2, q3, q4, q5;
    float t0, t1;
    float s0, s1, s2;

    p0 = two_prod(a.x, b.x, &q0);
    p1 = two_prod(a.x, b.y, &q1);
    p2 = two_prod(a.y, b.x, &q2);
    p3 = two_prod(a.x, b.z, &q3);
    p4 = two_prod(a.y, b.y, &q4);
    p5 = two_prod(a.z, b.x, &q5);

    three_sum(&p1, &p2, &q0);
    three_sum(&p2, &q1, &q2);
    three_sum(&p3, &p4, &p5);

    s0 = two_sum(p2, p3, &t0);
    s1 = two_sum(q1, p4, &t1);
    s2 = q2 + p5;
    s1 = two_sum(s1, t0, &t0);
    s2 += (t0 + t1);
    s1 += a.x * b.w + a.y * b.z + a.z * b.y + a.w * b.x + q0 + q3 + q4 + q5;

    renorm(&p0, &p1, &s0, &s1, &s2);

    (*prod).x = p0;
    (*prod).y = p1;
    (*prod).z = p2;
    (*prod).w = p3;
}

inline void qfMulFloat(float4 *prod, const float4 a, const float b)
{
    float p0, p1, p2, p3;
    float q0, q1, q2;
    float s0, s1, s2, s3, s4;

    p0 = two_prod(a.x, b, &q0);
    p1 = two_prod(a.y, b, &q1);
    p2 = two_prod(a.z, b, &q2);
    p3 = a.w * b;

    s0 = p0;
    s1 = two_sum(q0, p1, &s2);

    three_sum(&s2, &q1, &p2);
    three_sum2(&q1, &q2, &p3);

    s3 = q1;
    s4 = q2 + p2;

    renorm(&s0, &s1, &s2, &s3, &s4);

    (*prod).x = s0;
    (*prod).y = s1;
    (*prod).z = s2;
    (*prod).w = s3;
}

inline bool qfLessThan(float4 *a, float b)
{
    return ((*a).x < b || ((*a).x == b && (*a).y < 0.0f));
}

inline void renorm4(float *c0, float *c1,
                    float *c2, float *c3)
{
    float s0, s1, s2 = 0.0f, s3 = 0.0f;

    s0 = quick_two_sum(*c2, *c3, c3);
    s0 = quick_two_sum(*c1, s0, c2);
    *c0 = quick_two_sum(*c0, s0, c1);
    s0 = *c0;
    s1 = *c1;

    if (s1 != 0.0f)
    {
        s1 = quick_two_sum(s1, *c2, &s2);
        if (s2 != 0.0f)
        {
            s2 = quick_two_sum(s2, *c3, &s3);
        }
        else
        {
            s1 = quick_two_sum(s1, *c3, &s2);
        }
    }
    else
    {
        s0 = quick_two_sum(s0, *c2, &s1);
        if (s1 != 0.0f)
        {
            s1 = quick_two_sum(s1, *c3, &s2);
        }
        else
        {
            s0 = quick_two_sum(s0, *c3, &s1);
        }
    }

    *c0 = s0;
    *c1 = s1;
    *c2 = s2;
    *c3 = s3;
}

float4 qfDiv(const float4 a, const float4 b)
{

    float q0, q1, q2, q3;
    float4 r;
    float4 p;

    q0 = a.x / b.x;

    // r = a - (b * q0);
    qfMulFloat(&p, b, q0);
    p = qfNegate(p);
    qfAdd(&r, a, p);
    q1 = r.x / b.x;

    // r -= (b * q1);
    qfMulFloat(&p, b, q1);
    p = qfNegate(p);
    qfAdd(&r, r, p);
    q2 = r.x / b.x;

    //r -= (b * q2);
    qfMulFloat(&p, b, q2);
    p = qfNegate(p);
    qfAdd(&r, r, p);
    q3 = r.x / b.x;

    renorm4(&q0, &q1, &q2, &q3);

    return (float4)(q0, q1, q2, q3);
}

__kernel void compute(
    __global int *iters,
    __global double *lastValuesR,
    __global double *lastValuesI,
    __global double *distancesR,
    __global double *distancesI,
    int calcDistance,

    float4 xStart,
    float4 yStart,
    float4 xInc,
    float4 yInc,
    int maxIterations,
    double sqrEscapeRadius)
{

    const float escape = sqrEscapeRadius;

    const float4 x0 = xStart;
    const float4 y0 = yStart;
    const float4 dx = xInc;
    const float4 dy = yInc;

    float4 qfTemp = qfAssign(0);

    const float4 x = qfAssign(0);
    const float4 y = qfAssign(0);

    // x = x0 + xInc * X;
    qfMulFloat(&qfTemp, dx, X);
    qfAdd(&x, qfTemp, x0);

    // y = y0 + yInc * Y;
    qfMulFloat(&qfTemp, dy, Y);
    qfAdd(&y, qfTemp, y0);

    float4 zr = qfAssign(0);
    float4 zi = qfAssign(0);

    float4 zrSqr = qfAssign(0);
    float4 ziSqr = qfAssign(0);
    float4 magnitudeSquared = qfAssign(0);

    // distance
    float4 dr = (float4)(1);
    float4 di = (float4)(0);
    float4 new_dr;

    const bool tCalcDistance = calcDistance > 0;

    int iteration = 0;

    while (iteration < maxIterations && qfLessThan(&magnitudeSquared, escape))
    {
        if (tCalcDistance)
        {
            //         new_dr = 2.0f * (zr * dr - zi * di) + 1.0f;
            qfMul(&new_dr, zr, dr);
            qfMul(&qfTemp, zi, di);
            qfAdd(&new_dr, new_dr, qfNegate(qfTemp));
            qfMulFloat(&new_dr, new_dr, 2.0f);
            qfAdd(&new_dr, new_dr, (float4)(1.0f));

            //         di = 2.0f * (zr * di + zi * dr);
            qfMul(&di, zr, di);
            qfMul(&qfTemp, zi, dr);
            qfAdd(&di, di, qfTemp);
            qfMulFloat(&di, di, 2.0f);

            dr = new_dr;
        }

        // float xx = x*x;
        qfMul(&zrSqr, zr, zr);
        // float yy = y*y;
        qfMul(&ziSqr, zi, zi);

        //y = 2*zr*zi+i;
        qfMulFloat(&qfTemp, zr, 2);
        qfMul(&qfTemp, qfTemp, zi);
        qfAdd(&zi, qfTemp, y);

        //x = xx-yy+r;
        qfTemp.x = -ziSqr.x;
        qfTemp.y = -ziSqr.y;
        qfTemp.z = -ziSqr.z;
        qfTemp.w = -ziSqr.w;

        qfAdd(&qfTemp, zrSqr, qfTemp);
        qfAdd(&zr, qfTemp, x);

        qfAdd(&magnitudeSquared, zrSqr, ziSqr);

        iteration++;
    }

    const int tIndex = X + Y * WIDTH;
    iters[tIndex] = iteration;
    lastValuesR[tIndex] = (double)zr.x + (double)zr.y;
    lastValuesI[tIndex] = (double)zi.x + (double)zi.y;
    if (tCalcDistance)
    {
        distancesR[tIndex] = (double)dr.x + (double)dr.y;
        distancesI[tIndex] = (double)di.x + (double)di.y;
    }
}