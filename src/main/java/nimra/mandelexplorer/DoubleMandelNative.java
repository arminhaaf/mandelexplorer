package nimra.mandelexplorer;

import java.util.ArrayList;
import java.util.List;

/**
 * Created: 08.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class DoubleMandelNative extends AbstractDoubleMandelImpl implements MandelImplFactory {
    static {
        System.loadLibrary("mandel_jni");
    }

    private final Algo algo;

    public DoubleMandelNative() {
        this(Algo.Double);
    }

    public DoubleMandelNative(Algo pAlgo) {
        algo = pAlgo;
    }

    public static native void mandel(int pType, int[] pIters, double[] pLastZr, double[] pLastZi, double[] distancesR, double[] distancesI, int pMode,
            int pWidth, int pHeight, double pXStart, double pYStart, double juliaCr, double juliaCi, double pXInc, double pYInc, int pMaxIter, double pEscSqr);


    @Override
    public void mandel(final MandelParams pParams, final MandelResult pMandelResult, final Tile pTile) {
        // depends on the tile size of copy is needed

        // create tile arrays and copy them back
        final int tTileWidth = pTile.getWidth();
        final int tTileHeight = pTile.getHeight();
        int[] tItersTile = new int[tTileWidth * tTileHeight];
        double[] tLastZrTile = new double[tTileWidth * tTileHeight];
        double[] tLastZiTile = new double[tTileWidth * tTileHeight];
        double[] tDistanceRTile = new double[tTileWidth * tTileHeight];
        double[] tDistanceITile = new double[tTileWidth * tTileHeight];

        final double tXinc = getXinc(pParams, pMandelResult.width, pMandelResult.height);
        final double tYinc = getYinc(pParams, pMandelResult.width, pMandelResult.height);
        mandel(algo.code, tItersTile, tLastZrTile, tLastZiTile, tDistanceRTile, tDistanceITile,
               pParams.getCalcMode().getModeNumber(), tTileWidth, tTileHeight,
               getXmin(pParams, pMandelResult.width, pMandelResult.height) + pTile.startX * tXinc,
               getYmin(pParams, pMandelResult.width, pMandelResult.height) + pTile.startY * tYinc,
               0.0, 0.0,
               tXinc, tYinc, pParams.getMaxIterations(), getEscapeSqr(pParams));

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
            tMandelImpls.add(new DoubleMandelNative(tAlgo));
        }
        return tMandelImpls;
    }

    @Override
    public String toString() {
        return "Native " + algo.name;
    }

    @Override
    public boolean setComputeDevice(final ComputeDevice pDevice) {
        return pDevice == ComputeDevice.CPU;
    }

    public enum Algo {
        AVX2Double(1, "AVX2 Double"), AVX2Single(2, "AVX2 Single"), Double(3, "Double");

        int code;

        String name;

        Algo(final int pCode, final String pName) {
            code = pCode;
            name = pName;
        }
    }
}
