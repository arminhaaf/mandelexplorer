package nimra.mandelexplorer;

import nimra.mandelexplorer.util.NativeLoader;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

/**
 * Created: 08.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class DoubleMandelNative extends AbstractDoubleMandelImpl implements MandelImplFactory {
    private static final boolean NATIVE_LIB_LOADED;

    static {
        boolean tSuccess = false;
        try {
            tSuccess = NativeLoader.loadNativeLib("mandel_jni");
        } catch (Throwable ex) {
            ex.printStackTrace();
        }
        NATIVE_LIB_LOADED = tSuccess;
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
    public void mandel(final ComputeDevice pComputeDevice, final MandelParams pParams, final MandelResult pMandelResult, final Tile pTile) {
        // depends on the tile size of copy is needed

        // create tile arrays and copy them back
        final int tTileWidth = pTile.getWidth();
        final int tTileHeight = pTile.getHeight();
        final int[] tItersTile = new int[tTileWidth * tTileHeight];
        final double[] tLastZrTile = new double[tTileWidth * tTileHeight];
        final double[] tLastZiTile = new double[tTileWidth * tTileHeight];
        final double[] tDistanceRTile = new double[tTileWidth * tTileHeight];
        final double[] tDistanceITile = new double[tTileWidth * tTileHeight];

        final double tXinc = getXinc(pParams, pMandelResult.width, pMandelResult.height);
        final double tYinc = getYinc(pParams, pMandelResult.width, pMandelResult.height);
        mandel(algo.code, tItersTile, tLastZrTile, tLastZiTile, tDistanceRTile, tDistanceITile,
                pParams.getCalcMode().getModeNumber(), tTileWidth, tTileHeight,
                getXmin(pParams, pMandelResult.width, pMandelResult.height) + pTile.startX * tXinc,
                getYmin(pParams, pMandelResult.width, pMandelResult.height) + pTile.startY * tYinc,
                pParams.getJuliaCr().doubleValue(), pParams.getJuliaCi().doubleValue(),
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
    public boolean isAvailable() {
        return NATIVE_LIB_LOADED;
    }

    @Override
    public List<MandelImpl> getMandelImpls() {
        final List<MandelImpl> tMandelImpls = new ArrayList<>();
        if (NATIVE_LIB_LOADED) {
            for (Algo tAlgo : Algo.values()) {
                tMandelImpls.add(new DoubleMandelNative(tAlgo));
            }
        }
        return tMandelImpls;
    }

    @Override
    public boolean isPreciseFor(final BigDecimal pPixelSize) {
        return pPixelSize.compareTo(algo.pixelPrecision)>=0;
    }

    @Override
    public String toString() {
        return "Native " + algo.name;
    }

    @Override
    public boolean supports(final ComputeDevice pDevice) {
        return pDevice == ComputeDevice.CPU;
    }

    public enum Algo {
        AVX2Double(1, "AVX2 Double", new BigDecimal("3E-16")),
        AVX2Single(2, "AVX2 Single", new BigDecimal("1E-7")),
        Double(3, "Double", new BigDecimal("3E-16")),
        SSEDouble(3, "SSE2 Double", new BigDecimal("3E-16"));

        final int code;

        final String name;

        final BigDecimal pixelPrecision;

        Algo(final int pCode, final String pName, final BigDecimal pPixelPrecision) {
            code = pCode;
            name = pName;
            pixelPrecision = pPixelPrecision;
        }

    }
}
