package nimra.mandelexplorer;

import com.aparapi.Kernel;
import com.aparapi.Range;

/**
 * Created: 13.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class AparapiDoubleMandelImpl extends Kernel implements MandelImpl {

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
    protected int iters[];

    /**
     * buffer used to store the last calculated real value of a point -> used for some palette calculations
     */
    protected double lastValuesR[];

    /**
     * buffer used to store the last calculated imaginary value of a point -> used for some palette calculations
     */
    protected double lastValuesI[];

    protected double distancesR[];
    protected double distancesI[];

    // boolean is not available !?
    protected final boolean[] calcDistance = new boolean[1];


    public void mandel(final MandelParams pParams, final int pWidth, final int pHeight, final int pStartX, final int pEndX, final int pStartY, final int pEndY, final boolean pCalcDistance, final MandelResult pMandelResult) {
        xInc = pParams.getXInc(pWidth, pHeight).doubleValue();
        yInc = pParams.getYInc(pWidth, pHeight).doubleValue();
        xStart = pParams.getXMin(pWidth, pHeight).doubleValue() + pStartX * xInc;
        yStart = pParams.getYMin(pWidth, pHeight).doubleValue() + pStartY * yInc;
        escapeSqr = pParams.getEscapeRadius() * pParams.getEscapeRadius();

        maxIterations = pParams.getMaxIterations();

        width = pWidth;
        height = pHeight;


        tileWidth = pEndX - pStartX;
        tileHeight = pEndY - pStartY;

        // when to use smaller arrays to copy
        // system copies arrays to GPU and back, so it is better to copy smaller arrays ;-) However we must copy the smaller arrays to the image...
        final boolean tUseTileArrays = tileWidth * 2 < width;
        if (tUseTileArrays) {
            // create tile arrays and copy them back
            tileStartX = 0;
            tileStartY = 0;

            iters = new int[tileWidth * tileHeight];
            lastValuesR = new double[tileWidth * tileHeight];
            lastValuesI = new double[tileWidth * tileHeight];
            if (pCalcDistance) {
                distancesR = new double[tileWidth * tileHeight];
                distancesI = new double[tileWidth * tileHeight];
            } else {
                // minimal buffer to copy to GPU -> null buffer or zero size buffer is not allowed 
                distancesR = distancesI = new double[1];
            }
        } else {
            // work on the image buffer (copy large amount of memory to the GPU)
            tileStartX = pStartX;
            tileStartY = pStartY;
            tileWidth = width;
            tileHeight = height;

            iters = pMandelResult.iters;
            lastValuesR = pMandelResult.lastValuesR;
            lastValuesI = pMandelResult.lastValuesI;
            distancesR = pMandelResult.distancesR;
            distancesI = pMandelResult.distancesI;
        }

        final Range range = Range.create2D(pEndX - pStartX, pEndY - pStartY);
        execute(range);

        if (tUseTileArrays) {
            for (int y = 0; y < tileHeight; y++) {
                final int tDestPos = width * (pStartY + y) + pStartX;
                final int tSrcPos = y * tileWidth;
                System.arraycopy(iters, tSrcPos, pMandelResult.iters, tDestPos, tileWidth);
                System.arraycopy(lastValuesR, tSrcPos, pMandelResult.lastValuesR, tDestPos, tileWidth);
                System.arraycopy(lastValuesI, tSrcPos, pMandelResult.lastValuesI, tDestPos, tileWidth);
                if (pCalcDistance) {
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
            if (calcDistance[0]) {
                new_dr = 2.0 * (zr * dr - zi * di) + 1.0;
                di = 2.0 * (zr * di + zi * dr);
                dr = new_dr;
            }

            zi = (2 * zr * zi) + y;
            zr = (zrsqr - zisqr) + x;

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
        if (calcDistance[0]) {
            distancesR[tIndex] = dr;
            distancesI[tIndex] = di;
        }

    }

}