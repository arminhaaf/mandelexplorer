package nimra.mandelexplorer;

import com.aparapi.Kernel;
import com.aparapi.Range;

/**
 * Created: 08.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class MandelNative extends DoubleMandelImpl {
    static {
        System.loadLibrary("mandel_jni");
    }

    private final Algo algo;

    public MandelNative(Algo pAlgo, final int pWidth, final int pHeight) {
        super(pWidth, pHeight);

        algo = pAlgo;
    }

    public static native void mandel(int pType, int[] pIters, double[] pLastZr, double[] pLastZi, double[] distancesR, double[] distancesI, boolean pCalcDist,
            int pWidth, int pHeight, double pXStart, double pYStart, double pXInc, double pYInc, int pMaxIter, double pEscSqr);

    @Override public synchronized Kernel execute(Range pRange) {
        mandel(algo.code, iters, lastValuesR, lastValuesI, distancesR, distancesI, calcDistance[0], width, height, xStart, yStart, xInc, yInc, maxIterations, escapeSqr);
        return this;
    }

        @Override
    public void run() {

    }

    public enum Algo {
        AVX2Double(1), AVX2Single(2), Double(3);

        int code;

        Algo(final int pCode) {
            code = pCode;
        }
    }
}
