#include <stdio.h>

#define WIDTH gridDim.x*blockDim.x
#define HEIGHT gridDim.y*blockDim.y
#define X ((blockIdx.x * blockDim.x) + threadIdx.x)
#define Y ((blockIdx.y * blockDim.y) + threadIdx.y)

extern "C"
__global__ void computeFloat(
      int *iters,
      double4 area,
      int maxIterations,
      float sqrEscapeRadius
      ) {

   const float x = area.x + X*area.z;
   const float y = area.y + Y*area.w;
   const float cr = x;
   const float ci = y;


   const float escape = sqrEscapeRadius;

   float zr = x;
   float zi = y;
   float new_zr = 0.0f;

   int count = 0;
   for (; count<maxIterations; count++){
        const float zrsqr = zr * zr;
        const float zisqr = zi * zi;

        if ( (zrsqr + zisqr) >= escape ) {
            break;
        }

        new_zr = (zrsqr - zisqr) + cr;
        zi = ((2.0f * zr) * zi) + ci;
        zr = new_zr;

        //If in a periodic orbit, assume it is trapped
        if (zr == 0.0 && zi == 0.0) {
            count = maxIterations;
            break;
        }
   }
   const int tIndex = X + Y * WIDTH;
   iters[tIndex]  = count;
}



int main() {
    int tWidth = 160;
    int tHeight = 80;
    int tMaxIter = 100;

    double4 tArea = {-1.5,-1.0, 2.0/tWidth, 2.0/tHeight};

    int *tIters = (int*)malloc(tWidth*tHeight*sizeof(int));

    int *cuIters;

    cudaMalloc((void **)&cuIters, sizeof(int)*tWidth*tHeight);


    cudaMemcpy(cuIters, tIters ,tHeight*tWidth*sizeof(int) ,cudaMemcpyHostToDevice);

    int tThreadsX = 8;
    int tThreadsY = 4;

   dim3 blockSize = dim3( tThreadsX,tThreadsY);
   dim3 gridSize  = dim3( tWidth/tThreadsX, tHeight/tThreadsY );

    computeFloat<<<gridSize, blockSize>>>(
          cuIters,
          tArea,
          tMaxIter,
          2);

    cudaDeviceSynchronize();

    cudaMemcpy(tIters, cuIters ,tHeight*tWidth*sizeof(int) ,cudaMemcpyDeviceToHost);

    for ( int y=0;y<tHeight; y++) {
        for ( int x=0; x<tWidth; x++) {
            if ( tIters[x+y*tWidth]==tMaxIter) {
              printf("X");
            } else {
                printf(".");
            }
        }
        printf("\n");
    }

}

