package nimra.mandelexplorer;

import java.math.BigDecimal;

/**
 * Created: 15.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public abstract class AbstractDoubleMandelImpl implements MandelImpl {
    private BigDecimal pixelPrecision = new BigDecimal("3E-16");

    public void setPixelPrecision(final BigDecimal pPixelPrecision) {
        pixelPrecision = pPixelPrecision;
    }

    @Override
    public boolean isPreciseFor(final BigDecimal pPixelSize) {
        return pPixelSize.compareTo(pixelPrecision) >= 0;
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
}
