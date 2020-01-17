package nimra.mandelexplorer;

import java.math.BigDecimal;
import java.util.stream.IntStream;

/**
 * Created: 12.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class StreamParallelDoubleMandelImpl extends AbstractDoubleMandelImpl implements MandelImpl {

    @Override
    public boolean isPreciseFor(final BigDecimal pPixelSize) {
        return true;
    }

    @Override
    public void mandel(final ComputeDevice pComputeDevice, final MandelParams pParams, final MandelResult pMandelResult, final Tile pTile) {
        final double xmin = getXmin(pParams, pMandelResult.width, pMandelResult.height);
        final double ymin = getYmin(pParams, pMandelResult.width, pMandelResult.height);
        final double xinc = getXinc(pParams, pMandelResult.width, pMandelResult.height);
        final double yinc = getYinc(pParams, pMandelResult.width, pMandelResult.height);
        final double juliaCr = pParams.getJuliaCr().doubleValue();
        final double juliaCi = pParams.getJuliaCi().doubleValue();
        final double escapeSqr = getEscapeSqr(pParams);

        IntStream.range(pTile.startY, pTile.endY).parallel().forEach(y -> {
            final double tY = ymin + y * yinc;
            
            final double tCi = pParams.getCalcMode() == CalcMode.JULIA ? juliaCi : tY;
            for (int x = pTile.startX; x < pTile.endX; x++) {
                final double tX = xmin + x * xinc;
                final double tCr = pParams.getCalcMode() == CalcMode.JULIA ? juliaCr : tX;

                int count = 0;

                double zr = tX;
                double zi = tY;

                // cache the squares -> 10% faster
                double zrsqr = zr * zr;
                double zisqr = zi * zi;

                // distance
                double dr = 1;
                double di = 0;
                double new_dr;

                while ((count < pParams.getMaxIterations()) && ((zrsqr + zisqr) < escapeSqr)) {
                    if (pParams.getCalcMode() == CalcMode.MANDELBROT_DISTANCE) {
                        new_dr = 2.0 * (zr * dr - zi * di) + 1.0;
                        di = 2.0 * (zr * di + zi * dr);
                        dr = new_dr;
                    }

                    zi = (2 * zr * zi) + tCi;
                    zr = (zrsqr - zisqr) + tCr;

                    //If in a periodic orbit, assume it is trapped
                    if (zr == 0.0 && zi == 0.0) {
                        count = pParams.getMaxIterations();
                    } else {
                        zrsqr = zr * zr;
                        zisqr = zi * zi;
                        count++;
                    }
                }

                final int tIndex = y * pMandelResult.width + x;
                pMandelResult.iters[tIndex] = count;
                pMandelResult.lastValuesR[tIndex] = zr;
                pMandelResult.lastValuesI[tIndex] = zi;
                if (pParams.getCalcMode() == CalcMode.MANDELBROT_DISTANCE) {
                    pMandelResult.distancesR[tIndex] = dr;
                    pMandelResult.distancesI[tIndex] = di;
                }
            }
        });
    }

    @Override
    public boolean supports(final ComputeDevice pDevice) {
        return pDevice == ComputeDevice.CPU;
    }

    @Override
    public String toString() {
        return "Java Double";
    }
}
