package nimra.mandelexplorer.cuda;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUDA_MEMCPY2D;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmemorytype;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import nimra.mandelexplorer.AbstractDoubleMandelImpl;
import nimra.mandelexplorer.CalcMode;
import nimra.mandelexplorer.ComputeDevice;
import nimra.mandelexplorer.MandelImpl;
import nimra.mandelexplorer.MandelParams;
import nimra.mandelexplorer.MandelResult;
import nimra.mandelexplorer.Tile;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.Map;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuCtxPopCurrent;
import static jcuda.driver.JCudaDriver.cuCtxPushCurrent;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

/**
 * Created: 24.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class CudaDoubleMandelImpl extends AbstractDoubleMandelImpl implements MandelImpl {

    protected final Map<ComputeDevice, CudaContext> deviceContext = new HashMap<>();

    private String cudaResource;

    public CudaDoubleMandelImpl(String pCudaResource) {
        cudaResource = pCudaResource;
    }

    public CudaDoubleMandelImpl() {
        this("/cuda/DoubleMandel.ptx");
    }

    protected synchronized CudaContext getContext(final ComputeDevice pComputeDevice) {
        CudaContext tCudaContext = deviceContext.get(pComputeDevice);
        if (tCudaContext == null) {
            tCudaContext = prepareContext(pComputeDevice);
            deviceContext.put(pComputeDevice, tCudaContext);
        }
        return tCudaContext;
    }


    protected CudaContext prepareContext(ComputeDevice pComputeDevice) {
        CudaContext tCudaContext = new CudaContext();
        tCudaContext.context = new CUcontext();
        cuCtxCreate(tCudaContext.context, 0, ((CudaDevice)pComputeDevice.getDeviceDescriptor()).getCuDevice());

        try {
            Path tResourcePath = Paths.get(getClass().getResource(cudaResource).toURI());
            final Path tPtxFile = Files.createTempFile("ME", ".ptx");
            Files.copy(tResourcePath, tPtxFile, StandardCopyOption.REPLACE_EXISTING);
            // Load the ptx file.
            CUmodule tModule = new CUmodule();
            cuModuleLoad(tModule, tPtxFile.toString());

            // Obtain a function pointer to the "add" function.
            tCudaContext.function = new CUfunction();
            cuModuleGetFunction(tCudaContext.function, tModule, "compute");
        } catch (Exception pE) {
            pE.printStackTrace();
        }

        return tCudaContext;
    }

    @Override
    public boolean supports(final ComputeDevice pDevice) {
        return pDevice.getDeviceDescriptor() != null && pDevice.getDeviceDescriptor() instanceof CudaDevice;
    }


    @Override
    public void mandel(final ComputeDevice pComputeDevice, final MandelParams pParams, final MandelResult pMandelResult, final Tile pTile) {
        final CudaContext tCudaContext = getContext(pComputeDevice);
        cuCtxPushCurrent(tCudaContext.context);

        final CudaBuffers tCudaBuffers = new CudaBuffers(pParams, pTile);

        final double tXinc = pParams.getXInc(pMandelResult.width, pMandelResult.height).doubleValue();
        final double tYinc = pParams.getYInc(pMandelResult.width, pMandelResult.height).doubleValue();
        final double tXmin = getXmin(pParams, pMandelResult.width, pMandelResult.height) + pTile.startX * tXinc;
        final double tYmin = getYmin(pParams, pMandelResult.width, pMandelResult.height) + pTile.startY * tYinc;

        final Pointer tKernelParameters = Pointer.to(
                Pointer.to(tCudaBuffers.iters),
                Pointer.to(tCudaBuffers.lastR),
                Pointer.to(tCudaBuffers.lastI),
                Pointer.to(tCudaBuffers.distanceR),
                Pointer.to(tCudaBuffers.distanceI),
                Pointer.to(new int[]{pParams.getCalcMode().getModeNumber()}),
                Pointer.to(new int[]{pTile.startX, pTile.startY, pTile.getWidth(), pTile.getHeight()}),
                Pointer.to(new double[]{tXmin, tYmin, tXinc, tYinc}),
                Pointer.to(new double[]{pParams.getJuliaCr().doubleValue(), pParams.getJuliaCi().doubleValue()}),
                Pointer.to(new int[]{pParams.getMaxIterations()}),
                Pointer.to(new double[]{pParams.getEscapeRadius() * pParams.getEscapeRadius()})
        );


        // TODO warp size -> multiple of warpsize is the fastest -> this is device dependent
        final int blockSizeX = 8;
        final int blockSizeY = 8;
        final int gridSizeX = (pTile.getWidth() / blockSizeX) + 1;  // we need on grid more to have all pixels
        final int gridSizeY = (pTile.getHeight() / blockSizeY) + 1;
        cuLaunchKernel(tCudaContext.function,
                       gridSizeX, gridSizeY, 1,      // Grid dimension
                       blockSizeX, blockSizeY, 1,      // Block dimension
                       0, null,               // Shared memory size and stream
                       tKernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        // TODO -> read async (around 25 ms for 1024x1024 with distance)
        tCudaBuffers.readTo(pParams, pTile, pMandelResult);
        tCudaBuffers.free();

        cuCtxPopCurrent(tCudaContext.context);
    }

    /**
     * read tile into hostbuffer
     */
    protected static void readBuffer(final MandelResult pMandelResult, final Tile pTile, final CUdeviceptr pBuffer, Pointer pHostBuffer, int pDataTypeSize) {
        CUDA_MEMCPY2D tMEMCPY2D = new CUDA_MEMCPY2D();
        tMEMCPY2D.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_DEVICE;
        tMEMCPY2D.srcDevice = pBuffer;
        tMEMCPY2D.srcPitch = pTile.getWidth() * pDataTypeSize;

        tMEMCPY2D.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
        tMEMCPY2D.dstHost = pHostBuffer;
        tMEMCPY2D.dstXInBytes = pTile.startX * pDataTypeSize;
        tMEMCPY2D.dstY = pTile.startY;
        tMEMCPY2D.dstPitch = pMandelResult.width * pDataTypeSize;

        tMEMCPY2D.WidthInBytes = pTile.getWidth() * pDataTypeSize;
        tMEMCPY2D.Height = pTile.getHeight();

        JCudaDriver.cuMemcpy2D(tMEMCPY2D);
    }

    @Override
    public boolean isAvailable() {
        return CudaDevice.getDevices().size() > 0;
    }

    @Override
    public String toString() {
        return "Cuda Double";
    }

    protected static class CudaContext {

        public CUcontext context;
        public CUfunction function;

        private void cleanup() {
            if (context != null) {
                cuCtxDestroy(context);
                context = null;
                function = null;
            }
        }
    }

    protected static class CudaBuffers {

        final CUdeviceptr iters = new CUdeviceptr();
        final CUdeviceptr lastR = new CUdeviceptr();
        final CUdeviceptr lastI = new CUdeviceptr();
        final CUdeviceptr distanceR = new CUdeviceptr();
        final CUdeviceptr distanceI = new CUdeviceptr();


        public CudaBuffers(MandelParams pParams, Tile pTile) {

            final int tBufferSize = pTile.getHeight() * pTile.getWidth();
            final int tDistanceBufferSize = pParams.getCalcMode() == CalcMode.MANDELBROT_DISTANCE ? tBufferSize : 1;

            cuMemAlloc(iters, tBufferSize * Sizeof.INT);
            cuMemAlloc(lastR, tBufferSize * Sizeof.DOUBLE);
            cuMemAlloc(lastI, tBufferSize * Sizeof.DOUBLE);
            cuMemAlloc(distanceR, tDistanceBufferSize * Sizeof.DOUBLE);
            cuMemAlloc(distanceI, tDistanceBufferSize * Sizeof.DOUBLE);
        }

        public void readTo(MandelParams pMandelParams, Tile pTile, MandelResult pMandelResult) {
            readBuffer(pMandelResult, pTile, iters, Pointer.to(pMandelResult.iters), Sizeof.INT);
            readBuffer(pMandelResult, pTile, lastR, Pointer.to(pMandelResult.lastValuesR), Sizeof.DOUBLE);
            readBuffer(pMandelResult, pTile, lastI, Pointer.to(pMandelResult.lastValuesI), Sizeof.DOUBLE);

            if (pMandelParams.getCalcMode() == CalcMode.MANDELBROT_DISTANCE) {
                readBuffer(pMandelResult, pTile, distanceR, Pointer.to(pMandelResult.distancesR), Sizeof.DOUBLE);
                readBuffer(pMandelResult, pTile, distanceI, Pointer.to(pMandelResult.distancesI), Sizeof.DOUBLE);
            }
        }

        public void free() {
            cuMemFree(iters);
            cuMemFree(lastR);
            cuMemFree(lastI);
            cuMemFree(distanceR);
            cuMemFree(distanceI);
        }

    }

}
