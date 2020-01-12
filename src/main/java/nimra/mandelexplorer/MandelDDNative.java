package nimra.mandelexplorer;

import com.aparapi.Kernel;
import com.aparapi.Range;

/**
 * Created: 08.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class MandelDDNative extends DDMandelImpl {
    static {
        System.loadLibrary("mandel_jni");
    }

    private final Algo algo;

    public MandelDDNative(Algo pAlgo, final int pWidth, final int pHeight) {
        super(pWidth, pHeight);

        algo = pAlgo;
    }

    public static native void mandelDD(int pType, int[] pIters, double[] pLastZr, double[] pLastZi, double[] distancesR, double[] distancesI, boolean pCalcDist,
            int pWidth, int pHeight,
            double pXStartHi, double pXStartLo, double pYStartHi, double pYStartLo,
            double pXIncHi, double pXIncLo, double pYIncHi, double pYIncLo, int pMaxIter, double pEscSqr);

    @Override
    public synchronized Kernel execute(Range pRange) {
        System.out.println(xInc.getHi() + " " + xInc.getLo() + " - " + yInc.getHi() + " " + yInc.getLo());
        mandelDD(algo.code, iters, lastValuesR, lastValuesI, distancesR, distancesI, calcDistance[0], width, height,
                 xStart.getHi(), xStart.getLo(), yStart.getHi(), yStart.getLo(),
                 xInc.getHi(), xInc.getLo(), yInc.getHi(), yInc.getLo(), maxIterations, escapeSqr);
        return this;
    }

    @Override
    public void run() {

    }

    public enum Algo {
        AVXDoubleDouble(1),DoubleDouble(2);

        int code;

        Algo(final int pCode) {
            code = pCode;
        }
    }
}
