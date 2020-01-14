package nimra.mandelexplorer;

import java.math.BigDecimal;
import java.util.stream.IntStream;

/**
 * Created: 12.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class StreamParallelMandelImpl extends AbstractDoubleMandelImpl {

    @Override
    public boolean isPreciseFor(final BigDecimal pPixelSize) {
        return true;
    }

    @Override
    public void mandel(final MandelParams pParams,
            final int width, final int height, final int startX, final int endX, final int startY, final int endY,
            final Mode pMode,
            final MandelResult pMandelResult) {
        final double xmin = getXmin(pParams, width, height);
        final double ymin = getYmin(pParams, width, height);
        final double xinc = getXinc(pParams, width, height);
        final double yinc = getYinc(pParams, width, height);
        final double escapeSqr = getEscapeSqr(pParams);

        IntStream.range(startY, endY).parallel().forEach(y -> {
            final double ci = ymin + y * yinc;
            for (int x = startX; x < endX; x++) {
                final double cr = xmin + x * xinc;

                int count = 0;

                double zr = cr;
                double zi = ci;

                // cache the squares -> 10% faster
                double zrsqr = zr * zr;
                double zisqr = zi * zi;

                // distance
                double dr = 1;
                double di = 0;
                double new_dr;

                while ((count < pParams.getMaxIterations()) && ((zrsqr + zisqr) < escapeSqr)) {
                    if (pMode==Mode.MANDELBROT_DISTANCE) {
                        new_dr = 2.0 * (zr * dr - zi * di) + 1.0;
                        di = 2.0 * (zr * di + zi * dr);
                        dr = new_dr;
                    }

                    zi = (2 * zr * zi) + ci;
                    zr = (zrsqr - zisqr) + cr;

                    //If in a periodic orbit, assume it is trapped
                    if (zr == 0.0 && zi == 0.0) {
                        count = pParams.getMaxIterations();
                    } else {
                        zrsqr = zr * zr;
                        zisqr = zi * zi;
                        count++;
                    }
                }

                final int tIndex = y * width + x;
                pMandelResult.iters[tIndex] = count;
                pMandelResult.lastValuesR[tIndex] = zr;
                pMandelResult.lastValuesI[tIndex] = zi;
                if (pMode == Mode.MANDELBROT_DISTANCE) {
                    pMandelResult.distancesR[tIndex] = dr;
                    pMandelResult.distancesI[tIndex] = di;
                }
            }
        });
    }

    @Override
    public String toString() {
        return "Java Double";
    }
}
