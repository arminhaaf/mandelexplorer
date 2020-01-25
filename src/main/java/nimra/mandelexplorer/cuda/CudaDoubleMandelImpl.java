package nimra.mandelexplorer.cuda;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import nimra.mandelexplorer.AbstractDoubleMandelImpl;
import nimra.mandelexplorer.CalcMode;
import nimra.mandelexplorer.ComputeDevice;
import nimra.mandelexplorer.MandelImpl;
import nimra.mandelexplorer.MandelParams;
import nimra.mandelexplorer.MandelResult;
import nimra.mandelexplorer.Tile;

import java.nio.file.CopyOption;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;

import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

/**
 * Created: 24.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class CudaDoubleMandelImpl extends AbstractDoubleMandelImpl implements MandelImpl {

    private CUfunction function;

    public CudaDoubleMandelImpl(String pCudaResource) {

        try {
            Path tResourcePath = Paths.get(getClass().getResource(pCudaResource).toURI());
            final Path tPtxFile = Files.createTempFile("ME", ".ptx");
            Files.copy(tResourcePath, tPtxFile, StandardCopyOption.REPLACE_EXISTING);
            // Load the ptx file.
            CUmodule tModule = new CUmodule();
            cuModuleLoad(tModule, tPtxFile.toString());

            // Obtain a function pointer to the "add" function.
            function = new CUfunction();
            cuModuleGetFunction(function, tModule, "compute");
        } catch (Exception pE) {
            pE.printStackTrace();
        }

    }

    @Override
    public boolean supports(final ComputeDevice pDevice) {
        return pDevice.getDeviceDescriptor() != null && pDevice.getDeviceDescriptor() instanceof CudaDevice;
    }


    @Override
    public void mandel(final ComputeDevice pComputeDevice, final MandelParams pParams, final MandelResult pMandelResult, final Tile pTile) {

        final int tTileWidth = pTile.getWidth();
        final int tTileHeight = pTile.getHeight();
        final int tBufferSize = tTileHeight * tTileWidth;

        final int tDistanceBufferSize = pParams.getCalcMode() == CalcMode.MANDELBROT_DISTANCE ? tBufferSize : 1;


        CUdeviceptr tCudaIters = new CUdeviceptr();
        cuMemAlloc(tCudaIters, tBufferSize * Sizeof.INT);

        CUdeviceptr tCudaLastR = new CUdeviceptr();
        cuMemAlloc(tCudaLastR, tBufferSize * Sizeof.DOUBLE);

        CUdeviceptr tCudaLastI = new CUdeviceptr();
        cuMemAlloc(tCudaLastI, tBufferSize * Sizeof.DOUBLE);


        CUdeviceptr tCudaDistanceR = new CUdeviceptr();
        cuMemAlloc(tCudaDistanceR, tDistanceBufferSize * Sizeof.DOUBLE);

        CUdeviceptr tCudaDistanceI = new CUdeviceptr();
        cuMemAlloc(tCudaDistanceI, tDistanceBufferSize * Sizeof.DOUBLE);


        final double tXinc = pParams.getXInc(pMandelResult.width, pMandelResult.height).doubleValue();
        final double tYinc = pParams.getYInc(pMandelResult.width, pMandelResult.height).doubleValue();
        final double tXmin = getXmin(pParams, pMandelResult.width, pMandelResult.height) + pTile.startX * tXinc;
        final double tYmin = getYmin(pParams, pMandelResult.width, pMandelResult.height) + pTile.startY * tYinc;


//        int *iters,
//        double *lastValuesR,
//        double *lastValuesI,
//        double *distancesR,
//        double *distancesI,
//        int mode,
//        double4 area,
//        double2 julia,
//        int maxIterations,
//        double sqrEscapeRadius

        Pointer tKernelParameters = Pointer.to(
                Pointer.to(tCudaIters),
                Pointer.to(tCudaLastR),
                Pointer.to(tCudaLastI),
                Pointer.to(tCudaDistanceR),
                Pointer.to(tCudaDistanceI),
                Pointer.to(new int[]{pParams.getCalcMode().getModeNumber()}),
                Pointer.to(new double[]{tXmin, tYmin, tXinc, tYinc}),
                Pointer.to(new double[]{pParams.getJuliaCr().doubleValue(), pParams.getJuliaCi().doubleValue()}),
                Pointer.to(new int[]{pParams.getMaxIterations()}),
                Pointer.to(new double[]{pParams.getEscapeRadius() * pParams.getEscapeRadius()})
        );


        // Call the kernel function
        //         final long[] globalWorkSize = new long[2];
        //        globalWorkSize[0] = pTile.getWidth();
        //        globalWorkSize[1] = pTile.getHeight();
        //        .
        int blockSizeX = pTile.getWidth();
        int blockSizeY = pTile.getHeight();
        int gridSizeX = 1;
        cuLaunchKernel(function,
                       1, 1, 1,      // Grid dimension
                       blockSizeX, blockSizeY, 1,      // Block dimension
                       0, null,               // Shared memory size and stream
                       tKernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        // Keine Ahnung wie das zu lesen ist -> nur mit zwischenbuffer?
        int[] tItersBuffer = new int[tBufferSize];
        cuMemcpyDtoH(Pointer.to(tItersBuffer), tCudaIters,tBufferSize*Sizeof.INT);
        for ( int y=0; y<pTile.getHeight(); y++) {
            System.arraycopy(tItersBuffer, 0, pMandelResult.iters, pTile.startY*pMandelResult.width+pTile.startX, pTile.getWidth());
        }

//        readBuffer(pMandelResult, pTile, tCudaIters, Pointer.to(pMandelResult.iters), Sizeof.INT);

//
//            readBuffer(tOpenCLContext, pMandelResult, pTile, tCLiters, Pointer.to(pMandelResult.iters), Sizeof.cl_int);
//            readBuffer(tOpenCLContext, pMandelResult, pTile, tCLlastR, Pointer.to(pMandelResult.lastValuesR), Sizeof.cl_double);
//            readBuffer(tOpenCLContext, pMandelResult, pTile, tCLlastI, Pointer.to(pMandelResult.lastValuesI), Sizeof.cl_double);
//            if (pParams.getCalcMode() == CalcMode.MANDELBROT_DISTANCE) {
//                readBuffer(tOpenCLContext, pMandelResult, pTile, tCLdistanceR, Pointer.to(pMandelResult.distancesR), Sizeof.cl_double);
//                readBuffer(tOpenCLContext, pMandelResult, pTile, tCLdistanceI, Pointer.to(pMandelResult.distancesI), Sizeof.cl_double);
//            }
//
//            // TODO cache mem objects
        cuMemFree(tCudaIters);
        cuMemFree(tCudaLastR);
        cuMemFree(tCudaLastI);
        cuMemFree(tCudaDistanceR);
        cuMemFree(tCudaDistanceI);
    }


    /**
     * read tile into hostbuffer
     */
    protected void readBuffer(final MandelResult pMandelResult, final Tile pTile, final CUdeviceptr pBuffer, Pointer pHostBuffer, int pDataTypeSize) {
//        long[] bufferOffset = new long[]{0, 0, 0};
//        long[] hostOffset = new long[]{pTile.startX * pDataTypeSize, pTile.startY, 0};
//        long[] region = new long[]{pTile.getWidth() * pDataTypeSize, pTile.getHeight(), 1};
//        long bufferRowPitch = pTile.getWidth() * pDataTypeSize;
//        long bufferSlicePitch = 0;
//        long hostRowPitch = pMandelResult.width * pDataTypeSize;
//        long hostSlicePitch = 0;
//        JCuda.cudaMemcpy2D(pHostBuffer, pTile.startY * hostRowPitch + pTile.startX+pDataTypeSize,
//                           pBuffer, 0, pTile.getWidth() * pDataTypeSize, pTile.getHeight(), cudaMemcpyKind.cudaMemcpyDeviceToHost);
    }

//    public void setCode(final String pCode) {
//        code = pCode;
//    }
//
//    public String getCode() {
//        return code;
//    }
//
//    public String getCompilerOptions() {
//        return compilerOptions;
//    }
//
//    public void setCompilerOptions(final String pCompilerOptions) {
//        compilerOptions = pCompilerOptions;
//    }
//
//    @Override
//    public boolean isThreadSafe() {
//        return true;
//    }
//
//    @Override
//    public MandelImpl copy() {
//        return new CudaDoubleMandelImpl(this);
//    }

    @Override
    public boolean isAvailable() {
        return CudaDevice.getDevices().size() > 0;
    }
}
