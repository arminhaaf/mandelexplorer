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
public class FP128CLMandelKernel extends BDMandelKernel {

    private static final FP128CLMandel fp128CLMandel;

    static {
        FP128CLMandel tImpl = null;
        try {
            tImpl = ((OpenCLDevice)KernelManager.instance().bestDevice()).bind(FP128CLMandel.class);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        fp128CLMandel = tImpl;
    }


    public FP128CLMandelKernel(final int pWidth, final int pHeight) {
        super(pWidth, pHeight);
    }


    @Override
    public synchronized Kernel execute(Range pRange) {
        if (fp128CLMandel == null) {
            throw new RuntimeException("need open cl for " + this);
        }

        fp128CLMandel.computeMandelBrot(pRange, iters, lastValuesR, lastValuesI, distancesR, distancesI, calcDistance[0] ? 1 : 0,
                                        FP128.from(xStart).vec, FP128.from(yStart).vec, FP128.from(xInc).vec, FP128.from(yInc).vec,
                                        maxIterations, (int)escapeSqr);

        return this;
    }


    @Override
    public void run() {
        throw new RuntimeException("not used");
    }

    @Override
    public String toString() {
        return "FP128CL (problems on nvidia)";
    }

}

