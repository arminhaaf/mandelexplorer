package nimra.mandelexplorer;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.OpenCLDevice;
import com.aparapi.internal.kernel.KernelManager;

import java.math.BigDecimal;
import java.math.MathContext;

/**
 * Created: 31.12.19   by: Armin Haaf
 * <p>
 * A DoubleDouble implementation
 *
 * @author Armin Haaf
 */
public class FFCLMandelKernel extends MandelKernel {

    private static final FFCLMandel ffCLMandel;

    static {
        FFCLMandel tImpl = null;
        try {
            tImpl = ((OpenCLDevice) KernelManager.instance().bestDevice()).bind(FFCLMandel.class);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        ffCLMandel = tImpl;
    }


    /**
     * Maximum iterations we will check for.
     */
    private int maxIterations = 100;

    private BigDecimal xStart;
    private BigDecimal yStart;

    private BigDecimal xInc;
    private BigDecimal yInc;

    private double escapeSqr;

    private final BigDecimal BD_TWO = new BigDecimal(2);

    public FFCLMandelKernel(final int pWidth, final int pHeight) {
        super(pWidth, pHeight);
    }

    @Override
    public void init(final MandelParams pMandelParams) {
        maxIterations = pMandelParams.getMaxIterations();
        escapeSqr = pMandelParams.getEscapeRadius() * pMandelParams.getEscapeRadius();

        final BigDecimal tScaleX =  pMandelParams.getScale().multiply(new BigDecimal(width)).divide(new BigDecimal(height), MathContext.DECIMAL128);
        final BigDecimal tScaleY = pMandelParams.getScale();

        //xStart =  mandelParams.getX_Double() - tScaleX / 2.0;
        xStart =  pMandelParams.getX().subtract(tScaleX.divide(BD_TWO, MathContext.DECIMAL128));
        //yStart =  mandelParams.getY_Double() - tScaleY / 2.0;
        yStart =  pMandelParams.getY().subtract(tScaleY.divide(BD_TWO, MathContext.DECIMAL128));

        xInc = tScaleX.divide(new BigDecimal(width), MathContext.DECIMAL128);
        yInc = tScaleY.divide(new BigDecimal(height), MathContext.DECIMAL128);
    }

    private float[] convertToFF(BigDecimal pBigDecimal) {
        return convertToFF(pBigDecimal.doubleValue());
    }

    private float[] convertToFF(double pDouble) {
        float[] tFF = new float[2];

        tFF[0] = computeHi(pDouble);
        tFF[1] = computeLo(pDouble);
        return tFF;
    }

    private float computeLo(final double a) {
        final double temp = ((1 << 27) + 1) * a;
        final double hi = temp - (temp - a);
        final double lo = a - (float) hi;
        return (float) lo;
    }

    private float computeHi(final double a) {
        final double temp = ((1 << 27) + 1) * a;
        final double hi = temp - (temp - a);
        return (float) hi;
    }


    @Override
    public synchronized Kernel execute(Range pRange) {
        if (ffCLMandel == null) {
            throw new RuntimeException("need open cl for " + this);
        }

        ffCLMandel.computeMandelBrot(pRange, iters, lastValuesR, lastValuesI, distancesR, distancesI, calcDistance[0] ? 1 : 0,
                convertToFF(xStart), convertToFF(yStart), convertToFF(xInc), convertToFF(yInc), maxIterations, escapeSqr);

        return this;
    }


    @Override
    public void run() {
        throw new RuntimeException("not used");
    }

    @Override
    public String toString() {
        return "FFCL";
    }
}

