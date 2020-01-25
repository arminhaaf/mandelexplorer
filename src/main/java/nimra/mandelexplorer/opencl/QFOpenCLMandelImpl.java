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
public class QFOpenCLMandelImpl extends OpenCLMandelImpl {

    public QFOpenCLMandelImpl() {
        setCode(new Scanner(OpenCLDevice.class.getResourceAsStream("/opencl/QFMandel.cl"), "UTF-8").useDelimiter("\\A").next());
        setCompilerOptions(null);
    }

    private float[] convertToQF(BigDecimal pBigDecimal) {
        float[] tFF = new float[4];

        for (int i = 0; i < tFF.length; i++) {
            tFF[i] = (float)pBigDecimal.doubleValue();
            pBigDecimal = pBigDecimal.subtract(BigDecimal.valueOf(tFF[i]));
        }
        return tFF;
    }

    private float[] convertToQF(double pDouble) {
        float[] tFF = new float[4];

        tFF[0] = computeHi(pDouble);
        tFF[1] = computeLo(pDouble);
        return tFF;
    }

    private float computeLo(double a) {
        double temp = ((1 << 27) + 1) * a;
        double hi = temp - (temp - a);
        double lo = a - (float)hi;
        return (float)lo;
    }

    private float computeHi(double a) {
        double temp = ((1 << 27) + 1) * a;
        double hi = temp - (temp - a);
        return (float)hi;
    }


    @Override
    public void mandel(final ComputeDevice pComputeDevice, final MandelParams pParams, final MandelResult pMandelResult, final Tile pTile) {
        final OpenCLContext tOpenCLContext = getContext(pComputeDevice);

        synchronized ( tOpenCLContext) {
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
            final float[] tXinc = convertToQF(tBDXInc);
            final float[] tYinc = convertToQF(tBDYInc);
            final float[] tXmin = convertToQF(pParams.getXMin(pMandelResult.width, pMandelResult.height).add(tBDXInc.multiply(new BigDecimal(pTile.startX))));
            final float[] tYmin = convertToQF(pParams.getYMin(pMandelResult.width, pMandelResult.height).add(tBDYInc.multiply(new BigDecimal(pTile.startY))));

            clSetKernelArg(tOpenCLContext.kernel, 0, Sizeof.cl_mem, Pointer.to(tCLiters));
            clSetKernelArg(tOpenCLContext.kernel, 1, Sizeof.cl_mem, Pointer.to(tCLlastR));
            clSetKernelArg(tOpenCLContext.kernel, 2, Sizeof.cl_mem, Pointer.to(tCLlastI));
            clSetKernelArg(tOpenCLContext.kernel, 3, Sizeof.cl_mem, Pointer.to(tCLdistanceR));
            clSetKernelArg(tOpenCLContext.kernel, 4, Sizeof.cl_mem, Pointer.to(tCLdistanceI));
            clSetKernelArg(tOpenCLContext.kernel, 5, Sizeof.cl_uint, Pointer.to(new int[]{pParams.getCalcMode() == CalcMode.MANDELBROT_DISTANCE ? 1 : 0}));
            clSetKernelArg(tOpenCLContext.kernel, 6, Sizeof.cl_float4, Pointer.to(tXmin));
            clSetKernelArg(tOpenCLContext.kernel, 7, Sizeof.cl_float4, Pointer.to(tYmin));
            clSetKernelArg(tOpenCLContext.kernel, 8, Sizeof.cl_float4, Pointer.to(tXinc));
            clSetKernelArg(tOpenCLContext.kernel, 9, Sizeof.cl_float4, Pointer.to(tYinc));
            clSetKernelArg(tOpenCLContext.kernel, 10, Sizeof.cl_uint, Pointer.to(new int[]{pParams.getMaxIterations()}));
            clSetKernelArg(tOpenCLContext.kernel, 11, Sizeof.cl_double, Pointer.to(new double[]{pParams.getEscapeRadius() * pParams.getEscapeRadius()}));

            final long globalWorkSize[] = new long[2];
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
        return "QF OpenCL";
    }
}
