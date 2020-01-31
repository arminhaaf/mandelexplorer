#include <stdio.h>

#define WIDTH gridDim.x*blockDim.x
#define HEIGHT gridDim.y*blockDim.y
#define X ((blockIdx.x * blockDim.x) + threadIdx.x)
#define Y ((blockIdx.y * blockDim.y) + threadIdx.y)

extern "C"
__global__ void computeFloat(
      int *iters,
      float4 area,
      int maxIterations,
      float sqrEscapeRadius
      ) {

   const float cr = area.x + X*area.z;
   const float ci = area.y + Y*area.w;

   float zr = 0;
   float zi = 0;
   float new_zr = 0.0f;

   int count = 0;
   for (; count<maxIterations; count++){
        new_zr = (zr * zr - zi * zi) + cr;
        zi = ((2.0f * zr) * zi) + ci;
        zr = new_zr;

        if ( (zr * zr + zi * zi) >= sqrEscapeRadius ) {
            break;
        }
   }
   const int tIndex = X + Y * WIDTH;
   iters[tIndex]  = count;
}



int main() {
    int tWidth = 5000;
    int tHeight = 5000;
    int tMaxIter = 1000;

    float4 tArea = {-1.5f,-1.0f, 2.0f/tWidth, 2.0f/tHeight};

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
          4);
     cudaDeviceSynchronize();

    cudaMemcpy(tIters, cuIters ,tHeight*tWidth*sizeof(int) ,cudaMemcpyDeviceToHost);

    // 160 chars width and 80 chars height
    for ( int y=0;y<tHeight; y+=tHeight/80) {
        for ( int x=0; x<tWidth; x+=tWidth/160) {
            if ( tIters[x+y*tWidth]==tMaxIter) {
              printf("X");
            } else {
                printf(".");
            }
        }
        printf("\n");
    }

}

