package nimra.mandelexplorer.opencl;

import nimra.mandelexplorer.AbstractDoubleMandelImpl;
import nimra.mandelexplorer.CalcMode;
import nimra.mandelexplorer.ComputeDevice;
import nimra.mandelexplorer.MandelImpl;
import nimra.mandelexplorer.MandelParams;
import nimra.mandelexplorer.MandelResult;
import nimra.mandelexplorer.Tile;
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
import java.util.Map;

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

    private static final boolean OPENCL_AVAILABLE;

    static {
        boolean tOpenCLAvailable = false;
        try {
            CL.setExceptionsEnabled(true);
            tOpenCLAvailable = true;
        } catch (Throwable ex) {
            ex.printStackTrace();
        }
        OPENCL_AVAILABLE = tOpenCLAvailable;
    }

    private String code = "abcs";

    private String compilerOptions = "-cl-unsafe-math-optimizations -cl-fast-relaxed-math -cl-mad-enable -cl-finite-math-only";

    protected final Map<ComputeDevice, OpenCLContext> deviceContext = new HashMap<>();

    public OpenCLMandelImpl() {
    }

    public OpenCLMandelImpl(String pCode) {
        setCode(pCode);
    }

    public OpenCLMandelImpl(final OpenCLMandelImpl other) {
        this.code = other.code;
        this.compilerOptions = other.compilerOptions;
    }

    @Override
    public boolean supports(final ComputeDevice pDevice) {
        return pDevice.getDeviceDescriptor() != null && pDevice.getDeviceDescriptor() instanceof OpenCLDevice;
    }

    protected synchronized OpenCLContext getContext(final ComputeDevice pComputeDevice) {
        OpenCLContext tOpenCLContext = deviceContext.get(pComputeDevice);
        if (tOpenCLContext == null) {
            tOpenCLContext = prepareProgram((OpenCLDevice)pComputeDevice.getDeviceDescriptor());
            deviceContext.put(pComputeDevice, tOpenCLContext);
        }
        return tOpenCLContext;
    }


    protected OpenCLContext prepareProgram(final OpenCLDevice pDevice) {
        final OpenCLContext tOpenCLContext = new OpenCLContext();
        // Create a context for the selected device
        tOpenCLContext.context = clCreateContext(
                null, 1, new cl_device_id[]{pDevice.getCLDeviceId()},
                null, null, null);

        // Create the program
        tOpenCLContext.program = clCreateProgramWithSource(tOpenCLContext.context, 1,
                                                           new String[]{code}, null, null);

        // Build the program
        clBuildProgram(tOpenCLContext.program, 0, null, compilerOptions, null, null);

        // Create the kernel
        tOpenCLContext.kernel = clCreateKernel(tOpenCLContext.program, "compute", null);

        tOpenCLContext.commandQueue = clCreateCommandQueueWithProperties(
                tOpenCLContext.context, pDevice.getCLDeviceId(), new cl_queue_properties(), null);

        return tOpenCLContext;

    }

    @Override
    public void mandel(final ComputeDevice pComputeDevice, final MandelParams pParams, final MandelResult pMandelResult, final Tile pTile) {
        final OpenCLContext tOpenCLContext = getContext(pComputeDevice);

        // Kontext darf nur in einem Thread benutzt werden!
        synchronized (tOpenCLContext) {
            final double tXinc = pParams.getXInc(pMandelResult.width, pMandelResult.height).doubleValue();
            final double tYinc = pParams.getYInc(pMandelResult.width, pMandelResult.height).doubleValue();
            final double tXmin = getXmin(pParams, pMandelResult.width, pMandelResult.height) + pTile.startX * tXinc;
            final double tYmin = getYmin(pParams, pMandelResult.width, pMandelResult.height) + pTile.startY * tYinc;

            tOpenCLContext.prepareDefaultKernelBuffers(pParams, pTile);
            clSetKernelArg(tOpenCLContext.kernel, 6, Sizeof.cl_double4, Pointer.to(new double[]{tXmin, tYmin, tXinc, tYinc}));
            clSetKernelArg(tOpenCLContext.kernel, 7, Sizeof.cl_double2, Pointer.to(new double[]{pParams.getJuliaCr().doubleValue(), pParams.getJuliaCi().doubleValue()}));
            clSetKernelArg(tOpenCLContext.kernel, 8, Sizeof.cl_uint, Pointer.to(new int[]{pParams.getMaxIterations()}));
            clSetKernelArg(tOpenCLContext.kernel, 9, Sizeof.cl_double, Pointer.to(new double[]{pParams.getEscapeRadius() * pParams.getEscapeRadius()}));

            final long[] globalWorkSize = new long[2];
            globalWorkSize[0] = pTile.getWidth();
            globalWorkSize[1] = pTile.getHeight();


            clEnqueueNDRangeKernel(tOpenCLContext.commandQueue, tOpenCLContext.kernel, 2, null,
                                   globalWorkSize, null, 0, null, null);


            // TODO copy async -> about 70ms for 1024x1024 with distance
            tOpenCLContext.readTo(pParams, pTile, pMandelResult);
        }
    }

    /**
     * read tile into hostbuffer
     */
    protected static void readBuffer(final OpenCLContext pContext, final MandelResult pMandelResult, final Tile pTile, final cl_mem pBuffer, Pointer pHostBuffer, int pDataTypeSize) {
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

    @Override
    public boolean isAvailable() {
        return OPENCL_AVAILABLE;
    }

    protected static class OpenCLContext {

        public cl_context context;
        public cl_kernel kernel;
        public cl_command_queue commandQueue;
        public cl_program program;


        private int bufferSize;
        private CalcMode calcMode;

        private cl_mem iters;
        private cl_mem lastR;
        private cl_mem lastI;

        private cl_mem distanceR;
        private cl_mem distanceI;

        public cl_mem getIters() {
            return iters;
        }

        public cl_mem getLastR() {
            return lastR;
        }

        public cl_mem getLastI() {
            return lastI;
        }

        public cl_mem getDistanceR() {
            return distanceR;
        }

        public cl_mem getDistanceI() {
            return distanceI;
        }

        public void prepareDefaultKernelBuffers(MandelParams pMandelParams, Tile pTile) {
            createBuffers(pMandelParams, pTile);

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(iters));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(lastR));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(lastI));
            clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(distanceR));
            clSetKernelArg(kernel, 4, Sizeof.cl_mem, Pointer.to(distanceI));
            clSetKernelArg(kernel, 5, Sizeof.cl_uint, Pointer.to(new int[]{pMandelParams.getCalcMode().getModeNumber()}));
        }

        public void createBuffers(MandelParams pMandelParams, Tile pTile) {
            final int tTileWidth = pTile.getWidth();
            final int tTileHeight = pTile.getHeight();
            final int tBufferSize = tTileHeight * tTileWidth;

            if (bufferSize == tBufferSize && calcMode == pMandelParams.getCalcMode()) {
                return;
            }

            freeBuffers();

            final int tDistanceBufferSize = pMandelParams.getCalcMode() == CalcMode.MANDELBROT_DISTANCE ? tBufferSize : 1;

            iters = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                   tBufferSize * Sizeof.cl_int, null, null);
            lastR = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                   tBufferSize * Sizeof.cl_double, null, null);
            lastI = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                   tBufferSize * Sizeof.cl_double, null, null);

            distanceR = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                       tDistanceBufferSize * Sizeof.cl_double, null, null);
            distanceI = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                       tDistanceBufferSize * Sizeof.cl_double, null, null);

            bufferSize = tBufferSize;
        }


        public void readTo(MandelParams pParams, Tile pTile, MandelResult pMandelResult) {
            readBuffer(this, pMandelResult, pTile, iters, Pointer.to(pMandelResult.iters), Sizeof.cl_int);

            readBuffer(this, pMandelResult, pTile, lastR, Pointer.to(pMandelResult.lastValuesR), Sizeof.cl_double);

            readBuffer(this, pMandelResult, pTile, lastI, Pointer.to(pMandelResult.lastValuesI), Sizeof.cl_double);
            if (pParams.getCalcMode() == CalcMode.MANDELBROT_DISTANCE) {
                readBuffer(this, pMandelResult, pTile, distanceR, Pointer.to(pMandelResult.distancesR), Sizeof.cl_double);
                readBuffer(this, pMandelResult, pTile, distanceI, Pointer.to(pMandelResult.distancesI), Sizeof.cl_double);
            }
        }

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

            freeBuffers();
        }

        public void freeBuffers() {
            if (iters != null) {
                clReleaseMemObject(iters);
                iters = null;
            }

            if (lastR != null) {
                clReleaseMemObject(lastR);
                lastR = null;
            }

            if (lastI != null) {
                clReleaseMemObject(lastI);
                lastI = null;
            }
            if (distanceR != null) {
                clReleaseMemObject(distanceR);
                distanceR = null;
            }

            if (distanceI != null) {
                clReleaseMemObject(distanceI);
                distanceI = null;
            }
        }
    }

}
