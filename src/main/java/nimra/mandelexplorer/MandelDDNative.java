package nimra.mandelexplorer;

/**
 * Created: 08.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class MandelDDNative extends AbstractDDMandelImpl {
    static {
        System.loadLibrary("mandel_jni");
    }

    private final Algo algo;

    public MandelDDNative(Algo pAlgo) {
        algo = pAlgo;
    }

    public static native void mandelDD(int pType, int[] pIters, double[] pLastZr, double[] pLastZi, double[] distancesR, double[] distancesI, int pMode,
            int pWidth, int pHeight,
            double pXStartHi, double pXStartLo, double pYStartHi, double pYStartLo,
            double pJuliaCrHi, double pJuliaCrLo,double pJuliaCiHi, double pJuliaCiLo,
            double pXIncHi, double pXIncLo, double pYIncHi, double pYIncLo, int pMaxIter, double pEscSqr);

    @Override
    public void mandel(final MandelParams pParams, final int width, final int height, final int startX, final int endX, final int startY, final int endY, final Mode pMode, final MandelResult pMandelResult) {
        // create tile arrays and copy them back
        final int tTileWidth = endX - startX;
        final int tTileHeight = endY - startY;
        int[] tItersTile = new int[tTileWidth * tTileHeight];
        double[] tLastZrTile = new double[tTileWidth * tTileHeight];
        double[] tLastZiTile = new double[tTileWidth * tTileHeight];
        double[] tDistanceRTile = new double[tTileWidth * tTileHeight];
        double[] tDistanceITile = new double[tTileWidth * tTileHeight];

        final DD tXinc = getXinc(pParams, width, height);
        final DD tYinc = getYinc(pParams, width, height);
        final DD tXmin = getXmin(pParams, width, height).add(tXinc.multiply(startX));
        final DD tYmin = getYmin(pParams, width, height).add(tYinc.multiply(startY));
        mandelDD(algo.code, tItersTile, tLastZrTile, tLastZiTile, tDistanceRTile, tDistanceITile,
                 pMode.getModeNumber(), tTileWidth, tTileHeight,
                 tXmin.getHi(), tXmin.getLo(),
                 tYmin.getHi(), tYmin.getLo(),
                 0.0,0.0,0.0,0.0,
                 tXinc.getHi(), tXinc.getLo(),
                 tYinc.getHi(), tYinc.getLo(),
                 pParams.getMaxIterations(), getEscapeSqr(pParams));

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
        AVXDoubleDouble(1), DoubleDouble(2);

        private int code;

        private Algo(final int pCode) {
            code = pCode;
        }
    }
}
