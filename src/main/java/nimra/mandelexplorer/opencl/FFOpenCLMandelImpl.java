package nimra.mandelexplorer.opencl;

import nimra.mandelexplorer.CalcMode;
import nimra.mandelexplorer.ComputeDevice;
import nimra.mandelexplorer.MandelParams;
import nimra.mandelexplorer.MandelResult;
import nimra.mandelexplorer.Tile;
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
public class FFOpenCLMandelImpl extends OpenCLMandelImpl {

    public FFOpenCLMandelImpl() {
        setCode(new Scanner(OpenCLDevice.class.getResourceAsStream("/opencl/FFMandel.cl"), "UTF-8").useDelimiter("\\A").next());
        // compile should not optimize (no -cl-unsafe-math-optimizations !) -> however, seems that the optimization is always on
        // only Portable CPU implementation works as expected
        setCompilerOptions(null);
    }

    private float[] convertToFF(BigDecimal pBD) {
        final double tDouble = pBD.doubleValue();
        float[] tFF = new float[2];

        tFF[0] = computeHi(tDouble);
        tFF[1] = computeLo(tDouble);
        return tFF;
    }

    private float computeLo(final double a) {
        final double temp = ((1 << 27) + 1) * a;
        final double hi = temp - (temp - a);
        final double lo = a - (float)hi;
        return (float)lo;
    }

    private float computeHi(final double a) {
        final double temp = ((1 << 27) + 1) * a;
        final double hi = temp - (temp - a);
        return (float)hi;
    }

    @Override
    protected OpenCLContext prepareProgram(final OpenCLDevice pDevice) {
        if ( pDevice.getVendor().contains("Intel")) {
            // seems no intel side there are some compiler defaults which break FF
            setCompilerOptions("-cl-opt-disable -cl-finite-math-only -cl-mad-enable");
        } else {
            setCompilerOptions("-cl-fast-relaxed-math");
        }
        return super.prepareProgram(pDevice);
    }

    @Override
    public void mandel(final ComputeDevice pComputeDevice, final MandelParams pParams, final MandelResult pMandelResult, final Tile pTile) {
        final OpenCLContext tOpenCLContext = getContext(pComputeDevice);


        synchronized (tOpenCLContext) {

            final BigDecimal tBDXInc = pParams.getXInc(pMandelResult.width, pMandelResult.height);
            final BigDecimal tBDYInc = pParams.getYInc(pMandelResult.width, pMandelResult.height);
            final float[] tXinc = convertToFF(tBDXInc);
            final float[] tYinc = convertToFF(tBDYInc);
            final float[] tXmin = convertToFF(pParams.getXMin(pMandelResult.width, pMandelResult.height).add(tBDXInc.multiply(new BigDecimal(pTile.startX))));
            final float[] tYmin = convertToFF(pParams.getYMin(pMandelResult.width, pMandelResult.height).add(tBDYInc.multiply(new BigDecimal(pTile.startY))));
            final float[] tJuliaCr = convertToFF(pParams.getJuliaCr());
            final float[] tJuliaCi = convertToFF(pParams.getJuliaCi());

            tOpenCLContext.prepareDefaultKernelBuffers(pParams, pTile);
            clSetKernelArg(tOpenCLContext.kernel, 6, Sizeof.cl_float2, Pointer.to(tXmin));
            clSetKernelArg(tOpenCLContext.kernel, 7, Sizeof.cl_float2, Pointer.to(tYmin));
            clSetKernelArg(tOpenCLContext.kernel, 8, Sizeof.cl_float2, Pointer.to(tJuliaCr));
            clSetKernelArg(tOpenCLContext.kernel, 9, Sizeof.cl_float2, Pointer.to(tJuliaCi));
            clSetKernelArg(tOpenCLContext.kernel, 10, Sizeof.cl_float2, Pointer.to(tXinc));
            clSetKernelArg(tOpenCLContext.kernel, 11, Sizeof.cl_float2, Pointer.to(tYinc));
            clSetKernelArg(tOpenCLContext.kernel, 12, Sizeof.cl_uint, Pointer.to(new int[]{pParams.getMaxIterations()}));
            clSetKernelArg(tOpenCLContext.kernel, 13, Sizeof.cl_double, Pointer.to(new double[]{pParams.getEscapeRadius() * pParams.getEscapeRadius()}));

            final long globalWorkSize[] = new long[2];
            globalWorkSize[0] = pTile.getWidth();
            globalWorkSize[1] = pTile.getHeight();


            clEnqueueNDRangeKernel(tOpenCLContext.commandQueue, tOpenCLContext.kernel, 2, null,
                                   globalWorkSize, null, 0, null, null);


            tOpenCLContext.readTo(pParams,pTile,pMandelResult);
        }
    }

    @Override
    public String toString() {
        return "FF OpenCL";
    }
}
