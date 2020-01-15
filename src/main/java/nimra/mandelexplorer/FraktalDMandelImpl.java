package nimra.mandelexplorer;

import java.util.stream.IntStream;

/**
 * Created: 12.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class FraktalDMandelImpl implements MandelImpl {

    private final FractalDFunction fractalDFunction;

    public FraktalDMandelImpl(final FractalDFunction pFractalDFunction) {
        fractalDFunction = pFractalDFunction;
    }

    protected double getEscapeSqr(final MandelParams pParams) {
        return pParams.getEscapeRadius() * pParams.getEscapeRadius();
    }

    protected double getYinc(final MandelParams pParams, final int pWidth, final int pHeight) {
        return pParams.getYInc(pWidth, pHeight).doubleValue();
    }

    protected double getXinc(final MandelParams pParams, final int pWidth, final int pHeight) {
        return pParams.getXInc(pWidth, pHeight).doubleValue();
    }

    protected double getYmin(final MandelParams pParams, final int pWidth, final int pHeight) {
        return pParams.getYMin(pWidth, pHeight).doubleValue();
    }

    protected double getXmin(final MandelParams pParams, final int pWidth, final int pHeight) {
        return pParams.getXMin(pWidth, pHeight).doubleValue();
    }

    @Override
    public boolean supportsMode(final Mode pMode) {
        switch (pMode) {
            case MANDELBROT:
            case JULIA:
                return true;
            default:
                return false;
        }
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
        final double juliaCr = pParams.getJuliaCr().doubleValue();
        final double juliaCi = pParams.getJuliaCi().doubleValue();
        final double escapeSqr = getEscapeSqr(pParams);

        IntStream.range(startY, endY).parallel().forEach(y -> {
            final double tY = ymin + y * yinc;

            final ComplexD c = new ComplexD(0, pMode == Mode.JULIA ? juliaCi : tY);
            for (int x = startX; x < endX; x++) {
                final double tX = xmin + x * xinc;
                c.re = pMode == Mode.JULIA ? juliaCr : tX;

                int count = 0;

                final ComplexD z = new ComplexD(tX, tY);

                while ((count < pParams.getMaxIterations()) && (z.dist() < escapeSqr)) {


                    // z^3 + c
//                    double tmp = zrsqr * zr - 3 * zisqr * zr + tCr;
//                    zi = 3 * zrsqr * zi - zisqr * zi + tCi;
//                    zr = tmp;

                    fractalDFunction.calc(z, z, c);
                    count++;
                }

                final int tIndex = y * width + x;
                pMandelResult.iters[tIndex] = count;
                pMandelResult.lastValuesR[tIndex] = z.re;
                pMandelResult.lastValuesI[tIndex] = z.im;
            }
        });
    }


}
