package nimra.mandelexplorer;

import nimra.mandelexplorer.math.DD;

/**
 * Created: 12.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public abstract class AbstractDDMandelImpl implements MandelImpl {

    protected double getEscapeSqr(final MandelParams pParams) {
        return pParams.getEscapeRadius() * pParams.getEscapeRadius();
    }

    protected DD getYinc(final MandelParams pParams, final int pWidth, final int pHeight) {
        final DD tScaleY = new DD(pParams.getScale());

        return new DD(tScaleY).divide(pHeight);
    }

    protected DD getXinc(final MandelParams pParams, final int pWidth, final int pHeight) {
        final DD tScaleX = new DD(pParams.getScale()).multiply(pWidth).divide(pHeight);
        return new DD(tScaleX).divide(pWidth);
    }

    protected DD getYmin(final MandelParams pParams, final int pWidth, final int pHeight) {
        final DD tScaleY = new DD(pParams.getScale());
        return new DD(pParams.getY()).subtract(tScaleY.divide(2.0));
    }

    protected DD getXmin(final MandelParams pParams, final int pWidth, final int pHeight) {
        final DD tScaleX = new DD(pParams.getScale()).multiply(pWidth).divide(pHeight);
        return new DD(pParams.getX()).subtract(tScaleX.divide(2.0));
    }

}
