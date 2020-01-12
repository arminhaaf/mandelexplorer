package nimra.mandelexplorer;

/**
 * Created: 08.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class MandelNative extends AbstractDoubleMandelImpl {
    static {
        System.loadLibrary("mandel_jni");
    }

    private final Algo algo;

    public MandelNative(Algo pAlgo) {
        algo = pAlgo;
    }

    public static native void mandel(int pType, int[] pIters, double[] pLastZr, double[] pLastZi, double[] distancesR, double[] distancesI, boolean pCalcDist,
            int pWidth, int pHeight, double pXStart, double pYStart, double pXInc, double pYInc, int pMaxIter, double pEscSqr);


    @Override
    public void mandel(final MandelParams pParams, final int width, final int height, final int startX, final int endX, final int startY, final int endY, final boolean pCalcDistance, final MandelResult pMandelResult) {
        // create tile arrays and copy them back
        final int tTileWidth = endX - startX;
        final int tTileHeight = endY - startY;
        int[] tItersTile = new int[tTileWidth * tTileHeight];
        double[] tLastZrTile = new double[tTileWidth * tTileHeight];
        double[] tLastZiTile = new double[tTileWidth * tTileHeight];
        double[] tDistanceRTile = new double[tTileWidth * tTileHeight];
        double[] tDistanceITile = new double[tTileWidth * tTileHeight];

        final double tXinc = getXinc(pParams, width, height);
        final double tYinc = getYinc(pParams, width, height);
        mandel(algo.code, tItersTile, tLastZrTile, tLastZiTile, tDistanceRTile, tDistanceITile,
               pCalcDistance, tTileWidth, tTileHeight,
               getXmin(pParams, width, height) + startX * tXinc,
               getYmin(pParams, width, height) + startY * tYinc,
               tXinc, tYinc, pParams.getMaxIterations(), getEscapeSqr(pParams));

        for (int y = 0; y < tTileHeight; y++) {
            final int tDestPos = width * (startY + y) + startX;
            final int tSrcPos = y * tTileWidth;
            System.arraycopy(tItersTile, tSrcPos, pMandelResult.iters, tDestPos, tTileWidth);
            System.arraycopy(tLastZrTile, tSrcPos, pMandelResult.lastValuesR, tDestPos, tTileWidth);
            System.arraycopy(tLastZiTile, tSrcPos, pMandelResult.lastValuesI, tDestPos, tTileWidth);
            System.arraycopy(tDistanceRTile, tSrcPos, pMandelResult.distancesR, tDestPos, tTileWidth);
            System.arraycopy(tDistanceITile, tSrcPos, pMandelResult.distancesI, tDestPos, tTileWidth);
        }

    }


    public enum Algo {
        AVX2Double(1), AVX2Single(2), Double(3), VectorD(4);

        int code;

        Algo(final int pCode) {
            code = pCode;
        }
    }
}
