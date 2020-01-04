#define WIDTH get_global_size(0)
#define HEIGHT get_global_size(1)
#define X get_global_id(0)
#define Y get_global_id(1)

// Fixed-point 128 bits functions
// (c) EB Dec 2009

// We store signed 128-bit integers on one uint4.
// The associated real value is X/2^96 (ie first
// int is the integer part).

// Increment U
uint4 inc128(uint4 u)
{
  // Compute all carries to add
  int4 h = (u == (uint4)(0xFFFFFFFF)); // Note that == sets ALL bits if true (§6.3.d)
  uint4 c = (uint4)(h.y&h.z&h.w&1,h.z&h.w&1,h.w&1,1);
  return u+c;
}

// Return -U
uint4 neg128(uint4 u)
{
  return inc128(u ^ (uint4)(0xFFFFFFFF)); // 1 + not U
}

// Return representation of signed integer K
uint4 set128(int k)
{
  uint4 u = (uint4)((uint)abs(k),0,0,0);
  return (k<0)?neg128(u):u;
}

// Return U+V
uint4 add128(uint4 u,uint4 v)
{
  uint4 s = u+v;
  uint4 h = (uint4)(s.x < u.x ? 1U : 0U, s.y < u.y ? 1U : 0U,s.z < u.z ? 1U : 0U,s.w < u.w ? 1U : 0U);
  uint4 c1 = h.yzwx & (uint4)(1,1,1,0); // Carry from U+V
  h = (uint4)(s.x==0xFFFFFFFFU ? 1U : 0U, s.y ==0xFFFFFFFFU ? 1U : 0U,s.z ==0xFFFFFFFFU ? 1U : 0U,s.w ==0xFFFFFFFFU ? 1U : 0U);
  uint4 c2 = (uint4)((c1.y|(c1.z&h.z))&h.y,c1.z&h.z,0,0); // Propagated carry
  return s+c1+c2;
}

// Return U<<1
uint4 shl128(uint4 u)
{
  uint4 h = (u>>(uint4)(31U)) & (uint4)(0,1,1,1); // Bits to move up
  return (u<<(uint4)(1U)) | h.yzwx;
}

// Return U>>1
uint4 shr128(uint4 u)
{
  uint4 h = (u<<(uint4)(31U)) & (uint4)(0x80000000U,0x80000000U,0x80000000U,0); // Bits to move down
  return (u>>(uint4)(1U)) | h.wxyz;
}

// Return U*K.
// U MUST be positive.
uint4 mul128u(uint4 u,uint k)
{
  uint4 s1 = u * (uint4)(k);
  uint4 s2 = (uint4)(mul_hi(u.y,k),mul_hi(u.z,k),mul_hi(u.w,k),0);
  return add128(s1,s2);
}

// Return U*K, handles signs (K != INT_MIN).
uint4 mul128(uint4 u,int k)
{
  // Sign bits
  uint su = u.x & 0x80000000U;
  uint sk = (uint)k & 0x80000000U;
  // Abs values
  uint4 uu = (su)?neg128(u):u;
  uint kk = (uint)((sk)?(-k):k);
  // Product
  uint4 p = mul128u(uu,kk);
  // Product sign
  return (su^sk)?neg128(p):p;
}

// Return U*V truncated to keep the position of the decimal point.
// U and V MUST be positive.
uint4 mulfpu(uint4 u,uint4 v)
{
  // Diagonal coefficients
  uint4 s = (uint4)(u.x*v.x,mul_hi(u.y,v.y),u.y*v.y,mul_hi(u.z,v.z));
  // Off-diagonal
  uint4 t1 = (uint4)(mul_hi(u.x,v.y),u.x*v.y,mul_hi(u.x,v.w),u.x*v.w);
  uint4 t2 = (uint4)(mul_hi(v.x,u.y),v.x*u.y,mul_hi(v.x,u.w),v.x*u.w);
  s = add128(s,add128(t1,t2));
  t1 = (uint4)(0,mul_hi(u.x,v.z),u.x*v.z,mul_hi(u.y,v.w));
  t2 = (uint4)(0,mul_hi(v.x,u.z),v.x*u.z,mul_hi(v.y,u.w));
  s = add128(s,add128(t1,t2));
  t1 = (uint4)(0,0,mul_hi(u.y,v.z),u.y*v.z);
  t2 = (uint4)(0,0,mul_hi(v.y,u.z),v.y*u.z);
  s = add128(s,add128(t1,t2));
  // Add 3 to compensate for the truncation
  return add128(s,(uint4)(0,0,0,3));
}

// Return U*U truncated to keep the position of the decimal point.
// U MUST be positive.
uint4 sqrfpu(uint4 u)
{
  // Diagonal coefficients
  uint4 s = (uint4)(u.x*u.x,mul_hi(u.y,u.y),u.y*u.y,mul_hi(u.z,u.z));
  // Off-diagonal
  uint4 t = (uint4)(mul_hi(u.x,u.y),u.x*u.y,mul_hi(u.x,u.w),u.x*u.w);
  s = add128(s,shl128(t));
  t = (uint4)(0,mul_hi(u.x,u.z),u.x*u.z,mul_hi(u.y,u.w));
  s = add128(s,shl128(t));
  t = (uint4)(0,0,mul_hi(u.y,u.z),u.y*u.z);
  s = add128(s,shl128(t));
  // Add 3 to compensate for the truncation
  return add128(s,(uint4)(0,0,0,3));
}

// Return U*V, handles signs
uint4 mulfp(uint4 u,uint4 v)
{
  // Sign bits
  uint su = u.x & 0x80000000U;
  uint sv = v.x & 0x80000000U;
  // Abs values
  uint4 uu = (su)?neg128(u):u;
  uint4 vv = (sv)?neg128(v):v;
  // Product
  uint4 p = mulfpu(uu,vv);
  // Product sign
  return (su^sv)?neg128(p):p;
}

// Return U*U, handles signs
uint4 sqrfp(uint4 u)
{
  // Sign bit
  uint su = u.x & 0x80000000U;
  // Abs value
  uint4 uu = (su)?neg128(u):u;
  // Product is positive
  return sqrfpu(uu);
}

double shr(double v, int times) {
    while ( times-->0 ) {
        v = v/2;
    }
    return v;
}
double convert(uint4 v) {
    bool tNeg = (int)v.x<0;
    if ( (int)v.x<0 ) {
        v = neg128(v);
    }
    double tResult = v.x;

    tResult += shr(v.y,32);
    tResult += shr(v.z,64);
    tResult += shr(v.w,96);

    if ( tNeg ) {
    tResult *= -1;
    }
    return tResult;
}


__kernel void computeMandelBrot(
      __global uint *iters,
      __global double *lastValuesR,
      __global double *lastValuesI,
      __global double *distancesR,
      __global double *distancesI,
      int calcDistance,

      __global unsigned int* xStart,
      __global unsigned int* yStart,
      __global unsigned int* xInc,
      __global unsigned int* yInc,
      int maxIterations,
      int sqrEscapeRadius
      ) {

  // Convert inputs
  uint4 leftX = vload4(0,xStart);
  uint4 topY  = vload4(0,yStart);
  uint4 stepX = vload4(0,xInc);
  uint4 stepY = vload4(0,yInc);

    uint4 xc = add128(leftX,mul128(stepX,X)); // xc = leftX + xpix * stepX;
    uint4 yc = add128(topY,mul128(stepY,Y)); // yc = topY + ypix * stepY;

    vstore4(leftX, 0, iters);
    vstore4(topY, 1, iters);
    vstore4(stepX, 2, iters);
    vstore4(stepY, 3, iters);

    vstore4(add128(leftX, stepX), 4, iters);
    vstore4(add128(stepX, stepX), 5, iters);
    vstore4(mulfp(stepX, stepY), 6, iters);
    vstore4(mulfp(topY, stepY), 7, iters);

    vstore4(set128(X), 8, iters);
    vstore4(set128(Y), 9, iters);
    vstore4(xc, 10, iters);
    vstore4(yc, 11, iters);
    vstore4(mul128(stepX,14), 12, iters);
    vstore4(mul128(stepY,-19), 13, iters);
    vstore4(mulfp(set128(-1),set128(1)), 14, iters);
    vstore4(mulfp(set128(1),set128(-1)), 15, iters);
    vstore4(mulfp(set128(-1),set128(-1)), 16, iters);

    lastValuesR[0] = convert(leftX);
    lastValuesR[1] = convert(topY);
    lastValuesR[2] = convert(stepX);
    lastValuesR[3] = convert(stepY);

    lastValuesR[4] = convert(add128(leftX, stepX));
    lastValuesR[5] = convert(add128(stepX, stepX));
    lastValuesR[6] = convert(mulfp(stepX, stepY));
    lastValuesR[7] = convert(mulfp(topY, stepY));

    lastValuesR[8] = convert(set128(X));
    lastValuesR[9] = convert(set128(Y));
    lastValuesR[10] = convert(xc);
    lastValuesR[11] = convert(yc);
    lastValuesR[12] = convert(mul128(stepX,14));
    lastValuesR[13] = convert(mul128(stepY,-19));
    lastValuesR[14] = convert(mulfp(set128(-1),set128(1)));
    lastValuesR[15] = convert(mulfp(set128(1),set128(-1)));
    lastValuesR[16] = convert(mulfp(set128(-1),set128(-1)));

}