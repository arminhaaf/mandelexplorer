package nimra.mandelexplorer;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.OpenCLDevice;
import com.aparapi.internal.kernel.KernelManager;

import java.math.BigDecimal;

/**
 * Created: 31.12.19   by: Armin Haaf
 * <p>
 * A DoubleDouble implementation
 *
 * @author Armin Haaf
 */
public class QFCLMandelKernel extends BDMandelKernel {

    private static final QFCLMandel qfCLMandel;

    static {
        QFCLMandel tImpl = null;
        try {
            tImpl = ((OpenCLDevice)KernelManager.instance().bestDevice()).bind(QFCLMandel.class);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        qfCLMandel = tImpl;
    }


    public QFCLMandelKernel(final int pWidth, final int pHeight) {
        super(pWidth, pHeight);
    }

    private float[] convertToQF(BigDecimal pBigDecimal) {
        // how should this be done?
        return convertToQF(pBigDecimal.doubleValue());
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
        if (qfCLMandel == null) {
            throw new RuntimeException("need open cl for " + this);
        }

        qfCLMandel.computeMandelBrot(pRange, iters, lastValuesR, lastValuesI, distancesR, distancesI, calcDistance[0] ? 1 : 0,
                convertToQF(xStart), convertToQF(yStart), convertToQF(xInc), convertToQF(yInc), maxIterations, escapeSqr);

        return this;
    }


    @Override
    public void run() {
        throw new RuntimeException("not used");
    }

    @Override
    public String toString() {
        return "QFCL";
    }
}

