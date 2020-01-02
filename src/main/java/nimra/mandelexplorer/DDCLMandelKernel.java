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
public class DDCLMandelKernel extends DDMandelImpl {

    private static final DDCLMandel ddCLMandel;

    static {
        DDCLMandel tImpl = null;
        try {
            tImpl = ((OpenCLDevice) KernelManager.instance().bestDevice()).bind(DDCLMandel.class);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        ddCLMandel = tImpl;
    }


    public DDCLMandelKernel(final int pWidth, final int pHeight) {
        super(pWidth, pHeight);
    }



    private double[] convertToDD(DD pDD) {
        double[] tDD = new double[2];

        tDD[0] = pDD.getHi();
        tDD[1] = pDD.getLo();
        return tDD;
    }


    @Override
    public synchronized Kernel execute(Range pRange) {
        if (ddCLMandel == null) {
            throw new RuntimeException("need open cl for " + this);
        }

        ddCLMandel.computeMandelBrot(pRange, iters, lastValuesR, lastValuesI, distancesR, distancesI, calcDistance[0] ? 1 : 0,
                convertToDD(xStart), convertToDD(yStart), convertToDD(xInc), convertToDD(yInc), maxIterations, escapeSqr.getHi());

        return this;
    }


    @Override
    public void run() {
        throw new RuntimeException("not used");
    }

    @Override
    public String toString() {
        return "DDCL";
    }
}

