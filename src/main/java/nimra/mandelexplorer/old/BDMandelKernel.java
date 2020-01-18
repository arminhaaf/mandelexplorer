package nimra.mandelexplorer.old;

import nimra.mandelexplorer.MandelParams;

import java.math.BigDecimal;
import java.math.MathContext;

/**
 * Created: 31.12.19   by: Armin Haaf
 * <p>
 * A DoubleDouble implementation
 *
 * @author Armin Haaf
 */
public abstract class BDMandelKernel extends MandelKernel {

    protected int maxIterations = 100;

    protected BigDecimal xStart;
    protected BigDecimal yStart;

    protected BigDecimal xInc;
    protected BigDecimal yInc;

    protected double escapeSqr;

    protected final BigDecimal BD_0_5 = new BigDecimal("0.5");

    public BDMandelKernel(final int pWidth, final int pHeight) {
        super(pWidth, pHeight);
    }

    @Override
    public void init(final MandelParams pMandelParams) {
        maxIterations = pMandelParams.getMaxIterations();
        escapeSqr = pMandelParams.getEscapeRadius() * pMandelParams.getEscapeRadius();

        final BigDecimal tScaleX =  pMandelParams.getScale().multiply(new BigDecimal(width)).divide(new BigDecimal(height), MathContext.DECIMAL128);
        final BigDecimal tScaleY = pMandelParams.getScale();

        //xStart =  mandelParams.getX_Double() - tScaleX / 2.0;
        xStart =  pMandelParams.getX().subtract(tScaleX.multiply(BD_0_5));
        //yStart =  mandelParams.getY_Double() - tScaleY / 2.0;
        yStart =  pMandelParams.getY().subtract(tScaleY.multiply(BD_0_5));

        xInc = tScaleX.divide(new BigDecimal(width), MathContext.DECIMAL128);
        yInc = tScaleY.divide(new BigDecimal(height), MathContext.DECIMAL128);
    }

    public int getMaxIterations() {
        return maxIterations;
    }

    public BigDecimal getxStart() {
        return xStart;
    }

    public BigDecimal getyStart() {
        return yStart;
    }

    public BigDecimal getxInc() {
        return xInc;
    }

    public BigDecimal getyInc() {
        return yInc;
    }

    public double getEscapeSqr() {
        return escapeSqr;
    }

    @Override
    public void run() {
        throw new RuntimeException("not used");
    }

}

