package nimra.mandelexplorer;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.OpenCLDevice;
import com.aparapi.internal.kernel.KernelManager;

import java.math.BigDecimal;

/**
 * Created: 31.12.19   by: Armin Haaf
 * <p>
 *
 * QF Double implementation from https://github.com/gpu/JOCLSamples
 *
 * Should provide precision near to DoubleDouble implemention. However precision is only Double. Its unclear why, maybe QuadFloat implementation is broken
 * -> seems to be a compiler option -> -Ofast ist a problem
 *
 * @author Armin Haaf
 */
public class QFCLMandelKernel extends BDMandelKernel {

    public QFCLMandelKernel(final int pWidth, final int pHeight) {
        super(pWidth, pHeight);
    }

    private float[] convertToQF(BigDecimal pBigDecimal) {
        float[] tFF = new float[4];

        for ( int i=0; i<tFF.length; i++) {
            tFF[i] = (float)pBigDecimal.doubleValue();
            pBigDecimal = pBigDecimal.subtract(BigDecimal.valueOf(tFF[i]));
        }
        return tFF;
    }

    private float[] convertToQF(double pDouble) {
        float[] tFF = new float[4];

        tFF[0] = computeHi(pDouble);
        tFF[1] = computeLo(pDouble);
        return tFF;
    }

    private float computeLo(double a) {
        double temp = ((1<<27)+1) * a;
        double hi = temp - (temp - a);
        double lo = a - (float)hi;
        return (float)lo;
    }

    private float computeHi(double a) {
        double temp = ((1<<27)+1) * a;
        double hi = temp - (temp - a);
        return (float)hi;
    }

    @Override
    public synchronized Kernel execute(Range pRange) {
        QFCLMandel tImpl = CLImplCache.getImpl(this, QFCLMandel.class);

        tImpl.computeMandelBrot(pRange, iters, lastValuesR, lastValuesI, distancesR, distancesI, calcDistance[0] ? 1 : 0,
                convertToQF(xStart), convertToQF(yStart), convertToQF(xInc), convertToQF(yInc), maxIterations, escapeSqr);

        return this;
    }


    @Override
    public void run() {
        throw new RuntimeException("not used");
    }

    @Override
    public String toString() {
        return "QFCL (only double precision)";
    }
}

