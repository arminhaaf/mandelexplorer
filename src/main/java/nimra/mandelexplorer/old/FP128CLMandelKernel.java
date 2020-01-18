package nimra.mandelexplorer.old;

import com.aparapi.Kernel;
import com.aparapi.Range;
import nimra.mandelexplorer.FP128;

/**
 * Created: 31.12.19   by: Armin Haaf
 * <p>
 * A DoubleDouble implementation
 *
 * @author Armin Haaf
 */
public class FP128CLMandelKernel extends BDMandelKernel {
    public FP128CLMandelKernel(final int pWidth, final int pHeight) {
        super(pWidth, pHeight);
    }


    @Override
    public synchronized Kernel execute(Range pRange) {
        FP128CLMandel tImpl = CLImplCache.getImpl(this, FP128CLMandel.class);

        tImpl.computeMandelBrot(pRange, iters, lastValuesR, lastValuesI, distancesR, distancesI, calcDistance[0] ? 1 : 0,
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

