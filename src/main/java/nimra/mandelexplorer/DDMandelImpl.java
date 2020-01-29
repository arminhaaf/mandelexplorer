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
            // distance
            final DD dr = new DD();
            final DD di = new DD();
            final DD tTmpD1 = new DD();
            final DD tTmpD2 = new DD();
            final DD tX = new DD();
            final DD tCr = new DD();
            final DD zr = new DD();
            final DD zi = new DD();
            final DD zrsqr = new DD();
            final DD zisqr = new DD();

            final DD tY = new DD(yinc).selfMultiply(y).selfAdd(ymin);
            final DD tCi = pParams.getCalcMode() == CalcMode.JULIA ? juliaCi : tY;

            tX.setValue(xinc).selfMultiply(pTile.startX).selfAdd(xmin);
            for (int x = pTile.startX; x < pTile.endX; x++) {
                tCr.setValue(pParams.getCalcMode() == CalcMode.JULIA ? juliaCr : tX);

                int count = 0;

                zr.setValue(tX);
                zi.setValue(tY);

                // cache the squares -> 10% faster
                zrsqr.setValue(zr).selfSqr();
                zisqr.setValue(zi).selfSqr();

                dr.setValue(1);
                di.setValue(0);

                while ((count < pParams.getMaxIterations()) && ((zrsqr.hi + zisqr.hi) < escapeSqr)) {
                    if (pParams.getCalcMode() == CalcMode.MANDELBROT_DISTANCE) {
                        tTmpD2.setValue(zi).selfMultiply(di);
                        tTmpD1.setValue(zr).selfMultiply(dr).selfSubtract(tTmpD2).selfMultiply(2).selfAdd(1);
                        tTmpD2.setValue(zi).selfMultiply(dr);
                        di.selfMultiply(zr).selfAdd(tTmpD2).selfMultiply(2);
                        dr.setValue(tTmpD1);
                    }

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
                if (pParams.getCalcMode() == CalcMode.MANDELBROT_DISTANCE) {
                    pMandelResult.distancesR[tIndex] = dr.hi;
                    pMandelResult.distancesI[tIndex] = di.hi;
                }
                tX.selfAdd(xinc);
            }
        });
    }

    @Override
    public String toString() {
        return "DD Java";
    }
}
