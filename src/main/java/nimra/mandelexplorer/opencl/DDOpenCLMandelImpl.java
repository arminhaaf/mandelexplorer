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

    }

    private double[] convertToDD(BigDecimal pBD) {
        final DD tDD = new DD(pBD);

        return new double[]{tDD.getHi(), tDD.getLo()};
    }

    @Override
    public void mandel(final ComputeDevice pComputeDevice, final MandelParams pParams, final MandelResult pMandelResult, final Tile pTile) {
        final OpenCLContext tOpenCLContext = getContext(pComputeDevice);


        synchronized (tOpenCLContext) {
            final int tTileWidth = pTile.getWidth();
            final int tTileHeight = pTile.getHeight();
            final int tBufferSize = tTileHeight * tTileWidth;

            final int tDistanceBufferSize = pParams.getCalcMode() == CalcMode.MANDELBROT_DISTANCE ? tBufferSize : 1;

            final cl_mem tCLiters = clCreateBuffer(tOpenCLContext.context, CL_MEM_WRITE_ONLY,
                                                   tBufferSize * Sizeof.cl_int, null, null);
            final cl_mem tCLlastR = clCreateBuffer(tOpenCLContext.context, CL_MEM_WRITE_ONLY,
                                                   tBufferSize * Sizeof.cl_double, null, null);
            final cl_mem tCLlastI = clCreateBuffer(tOpenCLContext.context, CL_MEM_WRITE_ONLY,
                                                   tBufferSize * Sizeof.cl_double, null, null);

            final cl_mem tCLdistanceR;
            final cl_mem tCLdistanceI;
            if (pParams.getCalcMode() == CalcMode.MANDELBROT_DISTANCE) {
                tCLdistanceR = clCreateBuffer(tOpenCLContext.context, CL_MEM_WRITE_ONLY,
                                              tDistanceBufferSize * Sizeof.cl_double, null, null);
                tCLdistanceI = clCreateBuffer(tOpenCLContext.context, CL_MEM_WRITE_ONLY,
                                              tDistanceBufferSize * Sizeof.cl_double, null, null);
            } else {
                tCLdistanceR = clCreateBuffer(tOpenCLContext.context, CL_MEM_WRITE_ONLY,
                                              1, null, null);
                tCLdistanceI = clCreateBuffer(tOpenCLContext.context, CL_MEM_WRITE_ONLY,
                                              1, null, null);
            }

            final BigDecimal tBDXInc = pParams.getXInc(pMandelResult.width, pMandelResult.height);
            final BigDecimal tBDYInc = pParams.getYInc(pMandelResult.width, pMandelResult.height);
            final double[] tXinc = convertToDD(tBDXInc);
            final double[] tYinc = convertToDD(tBDYInc);
            final double[] tXmin = convertToDD(pParams.getXMin(pMandelResult.width, pMandelResult.height).add(tBDXInc.multiply(new BigDecimal(pTile.startX))));
            final double[] tYmin = convertToDD(pParams.getYMin(pMandelResult.width, pMandelResult.height).add(tBDYInc.multiply(new BigDecimal(pTile.startY))));
            final double[] tJuliaCr = convertToDD(pParams.getJuliaCr());
            final double[] tJuliaCi = convertToDD(pParams.getJuliaCi());

            clSetKernelArg(tOpenCLContext.kernel, 0, Sizeof.cl_mem, Pointer.to(tCLiters));
            clSetKernelArg(tOpenCLContext.kernel, 1, Sizeof.cl_mem, Pointer.to(tCLlastR));
            clSetKernelArg(tOpenCLContext.kernel, 2, Sizeof.cl_mem, Pointer.to(tCLlastI));
            clSetKernelArg(tOpenCLContext.kernel, 3, Sizeof.cl_mem, Pointer.to(tCLdistanceR));
            clSetKernelArg(tOpenCLContext.kernel, 4, Sizeof.cl_mem, Pointer.to(tCLdistanceI));
            clSetKernelArg(tOpenCLContext.kernel, 5, Sizeof.cl_uint, Pointer.to(new int[]{pParams.getCalcMode().getModeNumber()}));
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


            readBuffer(tOpenCLContext, pMandelResult, pTile, tCLiters, Pointer.to(pMandelResult.iters), Sizeof.cl_int);
            readBuffer(tOpenCLContext, pMandelResult, pTile, tCLlastR, Pointer.to(pMandelResult.lastValuesR), Sizeof.cl_double);
            readBuffer(tOpenCLContext, pMandelResult, pTile, tCLlastI, Pointer.to(pMandelResult.lastValuesI), Sizeof.cl_double);
            if (pParams.getCalcMode() == CalcMode.MANDELBROT_DISTANCE) {
                readBuffer(tOpenCLContext, pMandelResult, pTile, tCLdistanceR, Pointer.to(pMandelResult.distancesR), Sizeof.cl_double);
                readBuffer(tOpenCLContext, pMandelResult, pTile, tCLdistanceI, Pointer.to(pMandelResult.distancesI), Sizeof.cl_double);
            }

            // TODO cache mem objects
            clReleaseMemObject(tCLiters);
            clReleaseMemObject(tCLlastR);
            clReleaseMemObject(tCLlastI);
            clReleaseMemObject(tCLdistanceR);
            clReleaseMemObject(tCLdistanceI);
        }
    }

    @Override
    public String toString() {
        return "DD OpenCL";
    }
}
