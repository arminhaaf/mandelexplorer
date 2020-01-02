package nimra.mandelexplorer;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.OpenCLDevice;
import com.aparapi.internal.kernel.KernelManager;

/**
 * Created: 31.12.19   by: Armin Haaf
 * <p>
 * A DoubleDouble implementation
 *
 * @author Armin Haaf
 */
public class FloatCLMandelKernel extends FloatMandelKernel {
    private static final FloatCLMandel floatCLMandel;

    static {
        FloatCLMandel tImpl = null;
        try {
            tImpl = ((OpenCLDevice) KernelManager.instance().bestDevice()).bind(FloatCLMandel.class);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        floatCLMandel = tImpl;
    }

    public FloatCLMandelKernel(final int pWidth, final int pHeight) {
        super(pWidth, pHeight);
    }

    public synchronized Kernel execute(Range pRange) {
        if (floatCLMandel == null) {
            throw new RuntimeException("need open cl for " + this);
        }
        floatCLMandel.computeMandelBrot(pRange, iters, lastValuesR, lastValuesI, distancesR, distancesI, calcDistance[0] ? 1 : 0,
                xStart, yStart, xInc, yInc, maxIterations, escapeSqr);

        return this;
    }


    @Override
    public void run() {
        throw new RuntimeException("not used");
    }

    @Override
    public String toString() {
        return "FloatCL";
    }
}

