package nimra.mandelexplorer;

import nimra.mandelexplorer.math.DD;

import java.util.stream.IntStream;

/**
 * Created: 28.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class DDMandelImpl extends AbstractDDMandelImpl {

    @Override
    public boolean supports(final ComputeDevice pDevice) {
        return pDevice == ComputeDevice.CPU;
    }

    @Override
    public void mandel(final ComputeDevice pComputeDevice, final MandelParams pParams, final MandelResult pMandelResult, final Tile pTile) {
        final DD xmin = getXmin(pParams, pMandelResult.width, pMandelResult.height);
        final DD ymin = getYmin(pParams, pMandelResult.width, pMandelResult.height);
        final DD xinc = getXinc(pParams, pMandelResult.width, pMandelResult.height);
        final DD yinc = getYinc(pParams, pMandelResult.width, pMandelResult.height);
        final DD juliaCr = new DD(pParams.getJuliaCr());
        final DD juliaCi = new DD(pParams.getJuliaCi());
        final double escapeSqr = getEscapeSqr(pParams);

        IntStream.range(pTile.startY, pTile.endY).parallel().forEach(y -> {
            final DD tY = new DD(yinc).selfMultiply(y).selfAdd(ymin);
            final DD tCi = pParams.getCalcMode() == CalcMode.JULIA ? juliaCi : tY;
            for (int x = pTile.startX; x < pTile.endX; x++) {
                final DD tX = new DD(xinc).selfMultiply(x).selfAdd(xmin);
                final DD tCr = pParams.getCalcMode() == CalcMode.JULIA ? juliaCr : tX;

                int count = 0;

                final DD zr = new DD(tX);
                final DD zi = new DD(tY);

                // cache the squares -> 10% faster
                final DD zrsqr = new DD(zr).selfSqr();
                final DD zisqr = new DD(zi).selfSqr();

                while ((count < pParams.getMaxIterations()) && ((zrsqr.hi + zisqr.hi) < escapeSqr)) {

                    zi.selfMultiply(zr).selfMultiply(2).selfAdd(tCi);
                    zr.setValue(zrsqr).selfSubtract(zisqr).selfAdd(tCr);

                    zrsqr.setValue(zr).selfSqr();
                    zisqr.setValue(zi).selfSqr();
                    count++;
                }

                final int tIndex = y * pMandelResult.width + x;
                pMandelResult.iters[tIndex] = count;
                pMandelResult.lastValuesR[tIndex] = zr.hi;
                pMandelResult.lastValuesI[tIndex] = zi.hi;
            }
        });
    }

    @Override
    public String toString() {
        return "DD Java";
    }
}
