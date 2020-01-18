package nimra.mandelexplorer;

import java.util.ArrayList;
import java.util.List;

/**
 * Created: 08.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class DDMandelNative extends AbstractDDMandelImpl implements MandelImplFactory {
    static {
        System.loadLibrary("mandel_jni");
    }

    private final Algo algo;

    public DDMandelNative() {
        algo = Algo.DoubleDouble;
    }

    public DDMandelNative(Algo pAlgo) {
        algo = pAlgo;
    }

    public static native void mandelDD(int pType, int[] pIters, double[] pLastZr, double[] pLastZi, double[] distancesR, double[] distancesI, int pMode,
            int pWidth, int pHeight,
            double pXStartHi, double pXStartLo, double pYStartHi, double pYStartLo,
//            double pJuliaCrHi, double pJuliaCrLo, double pJuliaCiHi, double pJuliaCiLo,
            double pXIncHi, double pXIncLo, double pYIncHi, double pYIncLo, int pMaxIter, double pEscSqr);

    @Override
    public void mandel(final ComputeDevice pComputeDevice, final MandelParams pParams, final MandelResult pMandelResult, final Tile pTile) {
        // create tile arrays and copy them back
        final int tTileWidth = pTile.endX - pTile.startX;
        final int tTileHeight = pTile.endY - pTile.startY;
        int[] tItersTile = new int[tTileWidth * tTileHeight];
        double[] tLastZrTile = new double[tTileWidth * tTileHeight];
        double[] tLastZiTile = new double[tTileWidth * tTileHeight];
        double[] tDistanceRTile = new double[tTileWidth * tTileHeight];
        double[] tDistanceITile = new double[tTileWidth * tTileHeight];

        final DD tXinc = getXinc(pParams, pMandelResult.width, pMandelResult.height);
        final DD tYinc = getYinc(pParams, pMandelResult.width, pMandelResult.height);
        final DD tXmin = getXmin(pParams, pMandelResult.width, pMandelResult.height).add(tXinc.multiply(pTile.startX));
        final DD tYmin = getYmin(pParams, pMandelResult.width, pMandelResult.height).add(tYinc.multiply(pTile.startY));
        mandelDD(algo.code, tItersTile, tLastZrTile, tLastZiTile, tDistanceRTile, tDistanceITile,
                 pParams.getCalcMode().getModeNumber(), tTileWidth, tTileHeight,
                 tXmin.getHi(), tXmin.getLo(),
                 tYmin.getHi(), tYmin.getLo(),
//                 0.0, 0.0, 0.0, 0.0,
                 tXinc.getHi(), tXinc.getLo(),
                 tYinc.getHi(), tYinc.getLo(),
                 pParams.getMaxIterations(), getEscapeSqr(pParams));

        for (int y = 0; y < tTileHeight; y++) {
            final int tDestPos = pMandelResult.width * (pTile.startY + y) + pTile.startX;
            final int tSrcPos = y * tTileWidth;
            System.arraycopy(tItersTile, tSrcPos, pMandelResult.iters, tDestPos, tTileWidth);
            System.arraycopy(tLastZrTile, tSrcPos, pMandelResult.lastValuesR, tDestPos, tTileWidth);
            System.arraycopy(tLastZiTile, tSrcPos, pMandelResult.lastValuesI, tDestPos, tTileWidth);
            System.arraycopy(tDistanceRTile, tSrcPos, pMandelResult.distancesR, tDestPos, tTileWidth);
            System.arraycopy(tDistanceITile, tSrcPos, pMandelResult.distancesI, tDestPos, tTileWidth);
        }

    }

    @Override
    public List<MandelImpl> getMandelImpls() {
        final List<MandelImpl> tMandelImpls = new ArrayList<>();
        for (Algo tAlgo : Algo.values()) {
            tMandelImpls.add(new DDMandelNative(tAlgo));
        }
        return tMandelImpls;
    }

    @Override
    public String toString() {
        return "Native " + algo.name;
    }

    public enum Algo {
        AVXDoubleDouble(1, "AVX-DD"), DoubleDouble(2, "DD");

        final int code;

        final String name;

        Algo(final int pCode, final String pName) {
            code = pCode;
            name = pName;
        }
    }
}
