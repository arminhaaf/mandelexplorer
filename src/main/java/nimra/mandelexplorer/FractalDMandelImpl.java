package nimra.mandelexplorer;

import java.util.stream.IntStream;

/**
 * Created: 12.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class FractalDMandelImpl implements MandelImpl {

    private final FractalDFunction fractalDFunction;

    public FractalDMandelImpl(final FractalDFunction pFractalDFunction) {
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
    public boolean supportsMode(final CalcMode pMode) {
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
            final MandelResult pMandelResult, final Tile pTile) {
        final int tWidth = pMandelResult.width;
        final int tHeight = pMandelResult.height;
        final double xmin = getXmin(pParams, tWidth, tHeight);
        final double ymin = getYmin(pParams, tWidth, tHeight);
        final double xinc = getXinc(pParams, tWidth, tHeight);
        final double yinc = getYinc(pParams, tWidth, tHeight);
        final double juliaCr = pParams.getJuliaCr().doubleValue();
        final double juliaCi = pParams.getJuliaCi().doubleValue();
        final double escapeSqr = getEscapeSqr(pParams);

        IntStream.range(pTile.startY, pTile.endY).parallel().forEach(y -> {
            final double tY = ymin + y * yinc;

            final ComplexD c = new ComplexD(0, pParams.getCalcMode() == CalcMode.JULIA ? juliaCi : tY);
            for (int x = pTile.startX; x < pTile.endX; x++) {
                final double tX = xmin + x * xinc;
                c.re = pParams.getCalcMode() == CalcMode.JULIA ? juliaCr : tX;

                int count = 0;

                final ComplexD z = new ComplexD(tX, tY);

                while ((count < pParams.getMaxIterations()) && (z.magn() < escapeSqr)) {


                    // z^3 + c
//                    double tmp = zrsqr * zr - 3 * zisqr * zr + tCr;
//                    zi = 3 * zrsqr * zi - zisqr * zi + tCi;
//                    zr = tmp;

                    fractalDFunction.calc(z, z, c);
                    count++;
                }

                final int tIndex = y * tWidth + x;
                pMandelResult.iters[tIndex] = count;
                pMandelResult.lastValuesR[tIndex] = z.re;
                pMandelResult.lastValuesI[tIndex] = z.im;
            }
        });
    }


}
