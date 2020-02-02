package nimra.mandelexplorer.opencl;

import nimra.mandelexplorer.CalcMode;
import nimra.mandelexplorer.ComputeDevice;
import nimra.mandelexplorer.MandelParams;
import nimra.mandelexplorer.MandelResult;
import nimra.mandelexplorer.Tile;
import nimra.mandelexplorer.math.DD;
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
public class DDOpenCLMandelImpl extends OpenCLMandelImpl {

    public DDOpenCLMandelImpl() {
        setCode(new Scanner(OpenCLDevice.class.getResourceAsStream("/opencl/DDMandel.cl"), "UTF-8").useDelimiter("\\A").next());
        // compile should not optimize (no -cl-unsafe-math-optimizations !) -> however, seems that the optimization is always on
        // only Portable CPU implementation works as expected
        setCompilerOptions(null);

        setPixelPrecision(new BigDecimal("1E-32"));

    }

    private double[] convertToDD(BigDecimal pBD) {
        final DD tDD = new DD(pBD);

        return new double[]{tDD.getHi(), tDD.getLo()};
    }

    @Override
    public void mandel(final ComputeDevice pComputeDevice, final MandelParams pParams, final MandelResult pMandelResult, final Tile pTile) {
        final OpenCLContext tOpenCLContext = getContext(pComputeDevice);

        synchronized (tOpenCLContext) {

            final BigDecimal tBDXInc = pParams.getXInc(pMandelResult.width, pMandelResult.height);
            final BigDecimal tBDYInc = pParams.getYInc(pMandelResult.width, pMandelResult.height);
            final double[] tXinc = convertToDD(tBDXInc);
            final double[] tYinc = convertToDD(tBDYInc);
            final double[] tXmin = convertToDD(pParams.getXMin(pMandelResult.width, pMandelResult.height).add(tBDXInc.multiply(new BigDecimal(pTile.startX))));
            final double[] tYmin = convertToDD(pParams.getYMin(pMandelResult.width, pMandelResult.height).add(tBDYInc.multiply(new BigDecimal(pTile.startY))));
            final double[] tJuliaCr = convertToDD(pParams.getJuliaCr());
            final double[] tJuliaCi = convertToDD(pParams.getJuliaCi());

            tOpenCLContext.prepareDefaultKernelBuffers(pParams,pTile);
            clSetKernelArg(tOpenCLContext.kernel, 6, Sizeof.cl_double2, Pointer.to(tXmin));
            clSetKernelArg(tOpenCLContext.kernel, 7, Sizeof.cl_double2, Pointer.to(tYmin));
            clSetKernelArg(tOpenCLContext.kernel, 8, Sizeof.cl_double2, Pointer.to(tJuliaCr));
            clSetKernelArg(tOpenCLContext.kernel, 9, Sizeof.cl_double2, Pointer.to(tJuliaCi));
            clSetKernelArg(tOpenCLContext.kernel, 10, Sizeof.cl_double2, Pointer.to(tXinc));
            clSetKernelArg(tOpenCLContext.kernel, 11, Sizeof.cl_double2, Pointer.to(tYinc));
            clSetKernelArg(tOpenCLContext.kernel, 12, Sizeof.cl_uint, Pointer.to(new int[]{pParams.getMaxIterations()}));
            clSetKernelArg(tOpenCLContext.kernel, 13, Sizeof.cl_double, Pointer.to(new double[]{pParams.getEscapeRadius() * pParams.getEscapeRadius()}));

            final long[] globalWorkSize = new long[2];
            globalWorkSize[0] = pTile.getWidth();
            globalWorkSize[1] = pTile.getHeight();


            clEnqueueNDRangeKernel(tOpenCLContext.commandQueue, tOpenCLContext.kernel, 2, null,
                                   globalWorkSize, null, 0, null, null);


            tOpenCLContext.readTo(pParams, pTile,pMandelResult);
        }
    }

    @Override
    public String toString() {
        return "DD OpenCL";
    }
}
