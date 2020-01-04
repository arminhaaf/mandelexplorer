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
public class FFCLMandelKernel extends BDMandelKernel {


    public FFCLMandelKernel(final int pWidth, final int pHeight) {
        super(pWidth, pHeight);
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

        FFCLMandel tImpl = CLImplCache.getImpl(this, FFCLMandel.class);

        tImpl.computeMandelBrot(pRange, iters, lastValuesR, lastValuesI, distancesR, distancesI, calcDistance[0] ? 1 : 0,
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

