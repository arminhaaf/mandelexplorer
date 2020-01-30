package nimra.mandelexplorer;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.Device;
import com.aparapi.device.OpenCLDevice;

import java.util.List;

/**
 * Created: 13.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class AparapiDoubleMandelImpl extends Kernel implements MandelImpl {

    private static int MODE_JULIA = CalcMode.JULIA.getModeNumber();
    private static int MODE_DISTANCE = CalcMode.MANDELBROT_DISTANCE.getModeNumber();

    public int maxIterations = 100;

    protected double xStart;
    protected double yStart;

    protected double xInc;
    protected double yInc;

    protected double escapeSqr;


    protected int width;
    protected int height;

    protected int tileStartX;
    protected int tileStartY;
    protected int tileWidth;
    protected int tileHeight;

    /**
     * buffer used to store the iterations (width * height).
     */
    protected int[] iters;

    /**
     * buffer used to store the last calculated real value of a point -> used for some palette calculations
     */
    protected double[] lastValuesR;

    /**
     * buffer used to store the last calculated imaginary value of a point -> used for some palette calculations
     */
    protected double[] lastValuesI;

    protected double[] distancesR;
    protected double[] distancesI;

    protected double juliaCr;
    protected double juliaCi;

    // boolean is not available !?
    protected int mode;

    public void mandel(final ComputeDevice pComputeDevice, final MandelParams pParams, final MandelResult pMandelResult, final Tile pTile) {
        final int tWidth = pMandelResult.width;
        final int tHeight = pMandelResult.height;
        xInc = pParams.getXInc(tWidth, tHeight).doubleValue();
        yInc = pParams.getYInc(tWidth, tHeight).doubleValue();
        xStart = pParams.getXMin(tWidth, tHeight).doubleValue() + pTile.startX * xInc;
        yStart = pParams.getYMin(tWidth, tHeight).doubleValue() + pTile.startY * yInc;
        escapeSqr = pParams.getEscapeRadius() * pParams.getEscapeRadius();

        juliaCr = pParams.getJuliaCr().doubleValue();
        juliaCi = pParams.getJuliaCi().doubleValue();

        maxIterations = pParams.getMaxIterations();

        width = tWidth;
        height = tHeight;

        mode = pParams.getCalcMode().getModeNumber();

        tileWidth = pTile.getWidth();
        tileHeight = pTile.getHeight();

        // when to use smaller arrays to copy
        // system copies arrays to GPU and back, so it is better to copy smaller arrays ;-) However we must copy the smaller arrays to the image...
        final boolean tUseTileArrays = true;
        if (tUseTileArrays) {
            // create tile arrays and copy them back
            tileStartX = 0;
            tileStartY = 0;

            iters = new int[tileWidth * tileHeight];
            lastValuesR = new double[tileWidth * tileHeight];
            lastValuesI = new double[tileWidth * tileHeight];
            if (pParams.getCalcMode() == CalcMode.MANDELBROT_DISTANCE) {
                distancesR = new double[tileWidth * tileHeight];
                distancesI = new double[tileWidth * tileHeight];
            } else {
                // minimal buffer to copy to GPU -> null buffer or zero size buffer is not allowed 
                distancesR = distancesI = new double[1];
            }
        } else {
            // work on the image buffer (copy large amount of memory to the GPU)
            tileStartX = pTile.startX;
            tileStartY = pTile.startY;
            tileWidth = width;
            tileHeight = height;

            iters = pMandelResult.iters;
            lastValuesR = pMandelResult.lastValuesR;
            lastValuesI = pMandelResult.lastValuesI;
            distancesR = pMandelResult.distancesR;
            distancesI = pMandelResult.distancesI;
        }

        OpenCLDevice tDevice = null;
        if (pComputeDevice == ComputeDevice.CPU) {
            setExecutionMode(EXECUTION_MODE.JTP);
        } else if (pComputeDevice.getDeviceDescriptor() instanceof nimra.mandelexplorer.opencl.OpenCLDevice) {
            setExecutionMode(EXECUTION_MODE.GPU);
            final long tOpenCLDeviceId = ((nimra.mandelexplorer.opencl.OpenCLDevice)pComputeDevice.getDeviceDescriptor()).getDeviceId();
            final List<OpenCLDevice> tGPUDevices = OpenCLDevice.listDevices(Device.TYPE.GPU);
            for (OpenCLDevice tGPUDevice : tGPUDevices) {
                if (tGPUDevice.getDeviceId() == tOpenCLDeviceId) {
                    tDevice = tGPUDevice;
                    break;
                }
            }
        }

        final Range range = Range.create2D(tDevice, tileWidth, tileHeight);

        execute(range);

        if (tUseTileArrays) {
            for (int y = 0; y < tileHeight; y++) {
                final int tDestPos = width * (pTile.startY + y) + pTile.startX;
                final int tSrcPos = y * tileWidth;
                System.arraycopy(iters, tSrcPos, pMandelResult.iters, tDestPos, tileWidth);
                System.arraycopy(lastValuesR, tSrcPos, pMandelResult.lastValuesR, tDestPos, tileWidth);
                System.arraycopy(lastValuesI, tSrcPos, pMandelResult.lastValuesI, tDestPos, tileWidth);
                if (pParams.getCalcMode() == CalcMode.MANDELBROT_DISTANCE) {
                    System.arraycopy(distancesR, tSrcPos, pMandelResult.distancesR, tDestPos, tileWidth);
                    System.arraycopy(distancesI, tSrcPos, pMandelResult.distancesI, tDestPos, tileWidth);
                }
            }
        }
    }

    @Override
    public void run() {
        final int tX = getGlobalId(0);
        final int tY = getGlobalId(1);

        final double x = xStart + tX * xInc;
        final double y = yStart + tY * yInc;

        final double tCr = mode == MODE_JULIA ? juliaCr : x;
        final double tCi = mode == MODE_JULIA ? juliaCi : y;

        int count = 0;

        double zr = x;
        double zi = y;

        // cache the squares -> 10% faster
        double zrsqr = zr * zr;
        double zisqr = zi * zi;

        // distance
        double dr = 1;
        double di = 0;
        double new_dr;

        while ((count < maxIterations) && ((zrsqr + zisqr) < escapeSqr)) {
            if (mode == MODE_DISTANCE) {
                new_dr = 2.0 * (zr * dr - zi * di) + 1.0;
                di = 2.0 * (zr * di + zi * dr);
                dr = new_dr;
            }

            zi = (2 * zr * zi) + tCi;
            zr = (zrsqr - zisqr) + tCr;

            //If in a periodic orbit, assume it is trapped
            if (zr == 0.0 && zi == 0.0) {
                count = maxIterations;
            } else {
                zrsqr = zr * zr;
                zisqr = zi * zi;
                count++;
            }
        }

        final int tIndex = (tileStartY + tY) * tileWidth + tileStartX + tX;
        iters[tIndex] = count;
        lastValuesR[tIndex] = zr;
        lastValuesI[tIndex] = zi;
        if (mode == MODE_DISTANCE) {
            distancesR[tIndex] = dr;
            distancesI[tIndex] = di;
        }

    }

    @Override
    public boolean isThreadSafe() {
        return false;
    }

    @Override
    public MandelImpl copy() {
        return new AparapiDoubleMandelImpl();
    }

    @Override
    public boolean supports(final ComputeDevice pDevice) {
        return pDevice == ComputeDevice.CPU || pDevice.getDeviceDescriptor() instanceof nimra.mandelexplorer.opencl.OpenCLDevice;
    }

    @Override
    public void done() {
    }

    @Override
    public String toString() {
        return "Aparapi Double";
    }
}
