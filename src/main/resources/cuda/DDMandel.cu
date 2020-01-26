#define MODE_MANDEL 1
#define MODE_MANDEL_DISTANCE 2
#define MODE_JULIA 3

#define WIDTH gridDim.x*blockDim.x
#define HEIGHT gridDim.y*blockDim.y
#define X ((blockIdx.x * blockDim.x) + threadIdx.x)
#define Y ((blockIdx.y * blockDim.y) + threadIdx.y)

#define SPLIT  134217729.0 // 2^27+1, for IEEE double

__device__ inline double2 mul(const double2 pFF1, const double2 pFF2) {
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

    return make_double2(zhi, zlo);
}

__device__ inline double2 mulDouble(const double2 pFF1, const double pDouble) {
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

    return make_double2(zhi, zlo);
}


__device__ inline double2 add(const double2 pFF1, const double2 pFF2) {
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

    return make_double2(zhi, zlo);
}

__device__ inline double2 addDouble(const double2 pFF1, const double y) {
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

    return make_double2(hi, lo);
}

__device__ inline double2 sub(const double2 pFF1, const double2 pFF2) {
    return add(pFF1, make_double2(-pFF2.x, -pFF2.y));
}


extern "C"
__global__ void compute(
        int *iters,
        double *lastValuesR,
        double *lastValuesI,
        double *distancesR,
        double *distancesI,
        const int mode,
        const int4 tile,
        const double2 xStart,
        const double2 yStart,
        const double2 juliaCr,
        const double2 juliaCi,
        const double2 xInc,
        const double2 yInc,
        const int maxIterations,
        const double sqrEscapeRadius
) {

    if (X >= tile.z || Y >= tile.w) {      // tile.z is width of tile, tile.w is height of tile
        return;
    }


    const double2 x = add(make_double2(xStart.x, xStart.y), mulDouble(make_double2(xInc.x, xInc.y), X));
    const double2 y = add(make_double2(yStart.x, yStart.y), mulDouble(make_double2(yInc.x, yInc.y), Y));

    const double2 cr = mode == MODE_JULIA ? juliaCr : x;
    const double2 ci = mode == MODE_JULIA ? juliaCi : y;

    double2 zr = x;
    double2 zi = y;
    double2 tmp;

    // distance
    double2 dr = make_double2(1, 0);
    double2 di = make_double2(0, 0);
    double2 new_dr;

    int count = 0;

    for (; count < maxIterations; count++) {
        const double2 zrsqr = mul(zr, zr);
        const double2 zisqr = mul(zi, zi);

        if (add(zrsqr, zisqr).x >= sqrEscapeRadius) {
            break;
        }

        if (mode == MODE_MANDEL_DISTANCE) {
//            new_dr = 2.0f * (zr * dr - zi * di) + 1.0f;
            new_dr = addDouble(mulDouble(sub(mul(zr, dr), mul(zi, di)), 2.0), 1.0);
//            di = 2.0f * (zr * di + zi * dr);
            di = mulDouble(add(mul(zr, di), mul(zi, dr)), 2.0);
            dr = new_dr;
        }

        tmp = add(sub(zrsqr, zisqr), cr);
        zi = add(mulDouble(mul(zr, zi), 2.0), ci);
        zr = tmp;
    }

    const int tIndex = X + Y * tile.z;  // tile.z is width of tile
    iters[tIndex] = count;
    lastValuesR[tIndex] = (double) zr.x + (double) zr.y;
    lastValuesI[tIndex] = (double) zi.x + (double) zi.y;
    if (mode == MODE_MANDEL_DISTANCE) {
        distancesR[tIndex] = (double) dr.x + (double) dr.y;
        distancesI[tIndex] = (double) di.x + (double) di.y;
    }
}