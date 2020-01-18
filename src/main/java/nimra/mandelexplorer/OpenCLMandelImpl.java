package nimra.mandelexplorer;

import nimra.mandelexplorer.opencl.OpenCLDevice;
import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_program;
import org.jocl.cl_queue_properties;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import static org.jocl.CL.CL_MEM_WRITE_ONLY;
import static org.jocl.CL.clBuildProgram;
import static org.jocl.CL.clCreateBuffer;
import static org.jocl.CL.clCreateCommandQueueWithProperties;
import static org.jocl.CL.clCreateContext;
import static org.jocl.CL.clCreateKernel;
import static org.jocl.CL.clCreateProgramWithSource;
import static org.jocl.CL.clEnqueueNDRangeKernel;
import static org.jocl.CL.clEnqueueReadBufferRect;
import static org.jocl.CL.clReleaseCommandQueue;
import static org.jocl.CL.clReleaseContext;
import static org.jocl.CL.clReleaseKernel;
import static org.jocl.CL.clReleaseMemObject;
import static org.jocl.CL.clReleaseProgram;
import static org.jocl.CL.clSetKernelArg;

/**
 * Created: 14.01.20   by: Armin Haaf
 * <p>
 * Eine CL Implementierung wird immer nur eine Zeile, resp. einen Wert ermitteln. Nie ein Tile.
 *
 * @author Armin Haaf
 */
public class OpenCLMandelImpl extends AbstractDoubleMandelImpl implements MandelImpl {

    static {
        CL.setExceptionsEnabled(true);
    }

    private String code = "abcs";

    private String compilerOptions = "-cl-unsafe-math-optimizations -cl-fast-relaxed-math -cl-mad-enable -cl-finite-math-only";

    protected final Map<ComputeDevice, OpenCLContext> deviceContext = new HashMap<>();

    // Anzahl ermittelter werte -> bei float4 Implementierung 4,...
    // entsprechend müssen die buffers immer ein vielfaches von simdCount gross sein
    protected int simdCount = 1;

    public OpenCLMandelImpl() {
    }

    public OpenCLMandelImpl(String pCode) {
        setCode(pCode);
    }

    public OpenCLMandelImpl(final OpenCLMandelImpl other) {
        this.code = other.code;
        this.compilerOptions = other.compilerOptions;
        this.simdCount = other.simdCount;
    }

    @Override
    public boolean supports(final ComputeDevice pDevice) {
        return pDevice.getDeviceDescriptor() != null && pDevice.getDeviceDescriptor() instanceof OpenCLDevice;
    }

    public int getSimdCount() {
        return simdCount;
    }

    protected synchronized OpenCLContext getContext(final ComputeDevice pComputeDevice) {
        OpenCLContext tOpenCLContext = deviceContext.get(pComputeDevice);
        if (tOpenCLContext == null) {
            tOpenCLContext = prepareProgram((OpenCLDevice)pComputeDevice.getDeviceDescriptor());
            deviceContext.put(pComputeDevice, tOpenCLContext);
        }
        return tOpenCLContext;
    }


    private OpenCLContext prepareProgram(final OpenCLDevice pDevice) {
        final OpenCLContext tOpenCLContext = new OpenCLContext();
        // Create a context for the selected device
        tOpenCLContext.context = clCreateContext(
                null, 1, new cl_device_id[]{pDevice.getDeviceId()},
                null, null, null);

        // Create the program
        tOpenCLContext.program = clCreateProgramWithSource(tOpenCLContext.context, 1,
                                                           new String[]{code}, null, null);

        // Build the program
        clBuildProgram(tOpenCLContext.program, 0, null, compilerOptions, null, null);

        // Create the kernel
        tOpenCLContext.kernel = clCreateKernel(tOpenCLContext.program, "compute", null);

        tOpenCLContext.commandQueue = clCreateCommandQueueWithProperties(
                tOpenCLContext.context, pDevice.getDeviceId(), new cl_queue_properties(), null);

        return tOpenCLContext;

    }

    @Override
    public void mandel(final ComputeDevice pComputeDevice, final MandelParams pParams, final MandelResult pMandelResult, final Tile pTile) {
        final OpenCLContext tOpenCLContext = getContext(pComputeDevice);

        // Kontext darf nur in einem Thread benutzt werden!
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

            final double tXinc = pParams.getXInc(pMandelResult.width, pMandelResult.height).doubleValue();
            final double tYinc = pParams.getYInc(pMandelResult.width, pMandelResult.height).doubleValue();
            final double tXmin = getXmin(pParams, pMandelResult.width, pMandelResult.height) + pTile.startX * tXinc;
            final double tYmin = getYmin(pParams, pMandelResult.width, pMandelResult.height) + pTile.startY * tYinc;

            clSetKernelArg(tOpenCLContext.kernel, 0, Sizeof.cl_mem, Pointer.to(tCLiters));
            clSetKernelArg(tOpenCLContext.kernel, 1, Sizeof.cl_mem, Pointer.to(tCLlastR));
            clSetKernelArg(tOpenCLContext.kernel, 2, Sizeof.cl_mem, Pointer.to(tCLlastI));
            clSetKernelArg(tOpenCLContext.kernel, 3, Sizeof.cl_mem, Pointer.to(tCLdistanceR));
            clSetKernelArg(tOpenCLContext.kernel, 4, Sizeof.cl_mem, Pointer.to(tCLdistanceI));
            clSetKernelArg(tOpenCLContext.kernel, 5, Sizeof.cl_uint, Pointer.to(new int[]{pParams.getCalcMode() == CalcMode.MANDELBROT_DISTANCE ? 1 : 0}));
            clSetKernelArg(tOpenCLContext.kernel, 6, Sizeof.cl_double4, Pointer.to(new double[]{tXmin, tYmin, tXinc, tYinc}));
            clSetKernelArg(tOpenCLContext.kernel, 7, Sizeof.cl_uint, Pointer.to(new int[]{pParams.getMaxIterations()}));
            clSetKernelArg(tOpenCLContext.kernel, 8, Sizeof.cl_double, Pointer.to(new double[]{pParams.getEscapeRadius() * pParams.getEscapeRadius()}));

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

    /**
     * read tile into hostbuffer
     */
    protected void readBuffer(final OpenCLContext pContext, final MandelResult pMandelResult, final Tile pTile, final cl_mem pBuffer, Pointer pHostBuffer, int pDataTypeSize) {
        long[] bufferOffset = new long[]{0, 0, 0};
        long[] hostOffset = new long[]{pTile.startX * pDataTypeSize, pTile.startY, 0};
        long[] region = new long[]{pTile.getWidth() * pDataTypeSize, pTile.getHeight(), 1};
        long bufferRowPitch = pTile.getWidth() * pDataTypeSize;
        long bufferSlicePitch = 0;
        long hostRowPitch = pMandelResult.width * pDataTypeSize;
        long hostSlicePitch = 0;
        clEnqueueReadBufferRect(
                pContext.commandQueue, pBuffer, true, bufferOffset, hostOffset,
                region, bufferRowPitch, bufferSlicePitch, hostRowPitch,
                hostSlicePitch, pHostBuffer, 0, null, null);
    }

    public void setCode(final String pCode) {
        code = pCode;
    }

    public String getCode() {
        return code;
    }

    public String getCompilerOptions() {
        return compilerOptions;
    }

    public void setCompilerOptions(final String pCompilerOptions) {
        compilerOptions = pCompilerOptions;
    }

    @Override
    public boolean isThreadSafe() {
        return true;
    }

    @Override
    public MandelImpl copy() {
        return new OpenCLMandelImpl(this);
    }

    protected static class OpenCLContext {

        public cl_context context;
        public cl_kernel kernel;
        public cl_command_queue commandQueue;
        public cl_program program;


        private void cleanup() {
            if (kernel != null) {
                clReleaseKernel(kernel);
                kernel = null;
            }
            if (program != null) {
                clReleaseProgram(program);
                program = null;
            }

            if (commandQueue != null) {
                clReleaseCommandQueue(commandQueue);
                commandQueue = null;
            }
            if (context != null) {
                clReleaseContext(context);
                context = null;
            }
        }

    }
}
