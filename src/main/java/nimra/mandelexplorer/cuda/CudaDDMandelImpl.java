package nimra.mandelexplorer.cuda;

import jcuda.Pointer;
import nimra.mandelexplorer.ComputeDevice;
import nimra.mandelexplorer.MandelParams;
import nimra.mandelexplorer.MandelResult;
import nimra.mandelexplorer.Tile;
import nimra.mandelexplorer.math.DD;

import java.math.BigDecimal;

import static jcuda.driver.JCudaDriver.cuCtxPopCurrent;
import static jcuda.driver.JCudaDriver.cuCtxPushCurrent;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;

/**
 * Created: 25.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class CudaDDMandelImpl extends CudaDoubleMandelImpl {

    public CudaDDMandelImpl() {
        super("/cuda/DDMandel.ptx");
        setPixelPrecision(new BigDecimal("1E-32"));
    }

    private double[] convertToDD(BigDecimal pBD) {
        final DD tDD = new DD(pBD);

        return new double[]{tDD.getHi(), tDD.getLo()};
    }

    @Override
    public void mandel(final ComputeDevice pComputeDevice, final MandelParams pParams, final MandelResult pMandelResult, final Tile pTile) {
        final CudaContext tCudaContext = getContext(pComputeDevice);
        cuCtxPushCurrent(tCudaContext.context);

        CudaBuffers tCudaBuffers = new CudaBuffers(pParams, pTile);

        final BigDecimal tBDXInc = pParams.getXInc(pMandelResult.width, pMandelResult.height);
        final BigDecimal tBDYInc = pParams.getYInc(pMandelResult.width, pMandelResult.height);
        final double[] tXinc = convertToDD(tBDXInc);
        final double[] tYinc = convertToDD(tBDYInc);
        final double[] tXmin = convertToDD(pParams.getXMin(pMandelResult.width, pMandelResult.height).add(tBDXInc.multiply(new BigDecimal(pTile.startX))));
        final double[] tYmin = convertToDD(pParams.getYMin(pMandelResult.width, pMandelResult.height).add(tBDYInc.multiply(new BigDecimal(pTile.startY))));
        final double[] tJuliaCr = convertToDD(pParams.getJuliaCr());
        final double[] tJuliaCi = convertToDD(pParams.getJuliaCi());

        final Pointer tKernelParameters = Pointer.to(
                Pointer.to(tCudaBuffers.iters),
                Pointer.to(tCudaBuffers.lastR),
                Pointer.to(tCudaBuffers.lastI),
                Pointer.to(tCudaBuffers.distanceR),
                Pointer.to(tCudaBuffers.distanceI),
                Pointer.to(new int[]{pParams.getCalcMode().getModeNumber()}),
                Pointer.to(new int[]{pTile.startX, pTile.startY, pTile.getWidth(), pTile.getHeight()}),
                Pointer.to(tXmin),
                Pointer.to(tYmin),
                Pointer.to(tJuliaCr),
                Pointer.to(tJuliaCi),
                Pointer.to(tXinc),
                Pointer.to(tYinc),
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

    @Override
    public String toString() {
        return "Cuda DD";
    }

}
