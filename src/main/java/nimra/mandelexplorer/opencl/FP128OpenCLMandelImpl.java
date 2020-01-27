package nimra.mandelexplorer.opencl;

import nimra.mandelexplorer.CalcMode;
import nimra.mandelexplorer.ComputeDevice;
import nimra.mandelexplorer.MandelParams;
import nimra.mandelexplorer.MandelResult;
import nimra.mandelexplorer.Tile;
import nimra.mandelexplorer.math.FP128;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_mem;

import java.math.BigDecimal;
import java.util.Scanner;

import static org.jocl.CL.CL_MEM_WRITE_ONLY;
import static org.jocl.CL.clCreateBuffer;
import static org.jocl.CL.clEnqueueNDRangeKernel;
import static org.jocl.CL.clReleaseMemObject;
import static org.jocl.CL.clSetKernelArg;

/**
 * Created: 17.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class FP128OpenCLMandelImpl extends OpenCLMandelImpl {

    public FP128OpenCLMandelImpl() {
        setCode(new Scanner(OpenCLDevice.class.getResourceAsStream("/opencl/FP128Mandel.cl"), "UTF-8").useDelimiter("\\A").next());
        // compile should not optimize (no -cl-unsafe-math-optimizations !) -> however, seems that the optimization is always on
        // only Portable CPU implementation works as expected
        setCompilerOptions(null);

    }

    @Override
    public void mandel(final ComputeDevice pComputeDevice, final MandelParams pParams, final MandelResult pMandelResult, final Tile pTile) {
        final OpenCLContext tOpenCLContext = getContext(pComputeDevice);


        synchronized (tOpenCLContext) {
            final BigDecimal tBDXInc = pParams.getXInc(pMandelResult.width, pMandelResult.height);
            final BigDecimal tBDYInc = pParams.getYInc(pMandelResult.width, pMandelResult.height);
            final int[] tXinc = FP128.from(tBDXInc).vec;
            final int[] tYinc = FP128.from(tBDYInc).vec;
            final int[] tXmin = FP128.from(pParams.getXMin(pMandelResult.width, pMandelResult.height).add(tBDXInc.multiply(new BigDecimal(pTile.startX)))).vec;
            final int[] tYmin = FP128.from(pParams.getYMin(pMandelResult.width, pMandelResult.height).add(tBDYInc.multiply(new BigDecimal(pTile.startY)))).vec;
            final int[] tJuliaCr = FP128.from(pParams.getJuliaCr()).vec;
            final int[] tJuliaCi = FP128.from(pParams.getJuliaCi()).vec;

            tOpenCLContext.prepareDefaultKernelBuffers(pParams,pTile);
            clSetKernelArg(tOpenCLContext.kernel, 6, Sizeof.cl_uint4, Pointer.to(tXmin));
            clSetKernelArg(tOpenCLContext.kernel, 7, Sizeof.cl_uint4, Pointer.to(tYmin));
            clSetKernelArg(tOpenCLContext.kernel, 8, Sizeof.cl_uint4, Pointer.to(tJuliaCr));
            clSetKernelArg(tOpenCLContext.kernel, 9, Sizeof.cl_uint4, Pointer.to(tJuliaCi));
            clSetKernelArg(tOpenCLContext.kernel, 10, Sizeof.cl_uint4, Pointer.to(tXinc));
            clSetKernelArg(tOpenCLContext.kernel, 11, Sizeof.cl_uint4, Pointer.to(tYinc));
            clSetKernelArg(tOpenCLContext.kernel, 12, Sizeof.cl_uint, Pointer.to(new int[]{pParams.getMaxIterations()}));
            clSetKernelArg(tOpenCLContext.kernel, 13, Sizeof.cl_uint, Pointer.to(new int[]{(int)(pParams.getEscapeRadius() * pParams.getEscapeRadius())}));

            final long[] globalWorkSize = new long[2];
            globalWorkSize[0] = pTile.getWidth();
            globalWorkSize[1] = pTile.getHeight();


            clEnqueueNDRangeKernel(tOpenCLContext.commandQueue, tOpenCLContext.kernel, 2, null,
                                   globalWorkSize, null, 0, null, null);


            tOpenCLContext.readTo(pParams, pTile, pMandelResult);
        }
    }

    @Override
    public String toString() {
        return "Fixpoint 128 OpenCL";
    }
}
