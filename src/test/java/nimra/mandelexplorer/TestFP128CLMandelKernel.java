package nimra.mandelexplorer;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.OpenCLDevice;
import com.aparapi.internal.kernel.KernelManager;

/**
 * Created: 04.01.20   by: Armin Haaf
 *
 * Just for testing some opencl
 *
 * @author Armin Haaf
 */
public class TestFP128CLMandelKernel extends BDMandelKernel {

    private static final TestFP128CLMandel testFP128CLMandel;

    static {
        TestFP128CLMandel tImpl = null;
        try {
            tImpl = ((OpenCLDevice)KernelManager.instance().bestDevice()).bind(TestFP128CLMandel.class);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        testFP128CLMandel = tImpl;
    }


    public TestFP128CLMandelKernel(final int pWidth, final int pHeight) {
        super(pWidth, pHeight);
    }


    @Override
    public synchronized Kernel execute(Range pRange) {
        if (testFP128CLMandel == null) {
            throw new RuntimeException("need open cl for " + this);
        }

        testFP128CLMandel.computeMandelBrot(pRange, iters, lastValuesR, lastValuesI, distancesR, distancesI, calcDistance[0] ? 1 : 0,
                                            FP128.from(xStart).vec, FP128.from(yStart).vec, FP128.from(xInc).vec, FP128.from(yInc).vec,
                                            maxIterations, (int)escapeSqr);

        return this;
    }


    @Override
    public void run() {
        throw new RuntimeException("not used");
    }

}
