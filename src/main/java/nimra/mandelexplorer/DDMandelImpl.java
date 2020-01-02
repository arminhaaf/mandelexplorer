package nimra.mandelexplorer;

/**
 * Created: 31.12.19   by: Armin Haaf
 * <p>
 * A DoubleDouble implementation
 *
 * @author Armin Haaf
 */
public class DDMandelImpl extends MandelKernel {

    private int maxIterations = 100;

    private DD xStart;
    private DD yStart;

    private DD xInc;
    private DD yInc;

    private DD escapeSqr;

    public DDMandelImpl(final int pWidth, final int pHeight) {
        super(pWidth, pHeight);
    }


    @Override
    public void init(final MandelParams pMandelParams) {
        maxIterations = pMandelParams.getMaxIterations();

        double tScaleX = pMandelParams.getScale_double() * (width / (double) height);
        double tScaleY = pMandelParams.getScale_double();
        xStart = new DD(pMandelParams.getX_Double() - tScaleX / 2.0);
        yStart = new DD(pMandelParams.getY_Double() - tScaleY / 2.0);
        xInc = new DD(tScaleX/(double)width);
        yInc = new DD(tScaleY/(double)height);

        escapeSqr = new DD(pMandelParams.getEscapeRadius() * pMandelParams.getEscapeRadius());
    }

    public void run() {
        final int tX = getGlobalId(0);
        final int tY = getGlobalId(1);

        final DD x = xInc.multiply(tX).selfAdd(xStart);
        final DD y = yInc.multiply(tY).selfAdd(yStart);

        int count = 0;

        final DD zr = new DD(x);
        final DD zi = new DD(y);

//        // Iterate until the algorithm converges or until maxIterations are reached.
//        while ((count < maxIterations) && (((zx * zx) + (zy * zy)) < 8)) {
//            new_zx = ((zx * zx) - (zy * zy)) + x;
//            zy = (2 * zx * zy) + y;
//            zx = new_zx;
//            count++;
//        }
//

        // cache the squares -> 10% faster
        final DD zrsqr = zr.sqr();
        final DD zisqr = zi.sqr();

        // distance
        final DD dr = new DD(1);
        final DD di = new DD(0);

        final DD tmpDD = new DD();

        while ((count < maxIterations) && ((zrsqr.add(zisqr)).lt(escapeSqr))) {

            if (calcDistance[0]) {
                tmpDD.setValue(zr).selfMultiply(dr).selfSubtract(zi.multiply(di)).selfMultiply(2.0).selfAdd(1.0);
                di.selfMultiply(zr).selfAdd(zi.multiply(dr)).selfMultiply(2.0);
                dr.setValue(tmpDD);
            }

            tmpDD.setValue(zrsqr).selfSubtract(zisqr).selfAdd(x);
            zi.selfMultiply(zr).selfMultiply(2.0).selfAdd(y);
            zr.setValue(tmpDD);

            //If in a periodic orbit, assume it is trapped
            if (zr.isZero() && zi.isZero()) {
                count = maxIterations;
            } else {
                zrsqr.setValue(zr).selfSqr();
                zisqr.setValue(zi).selfSqr();
                count++;
            }
        }

        final int tIndex = tY * getGlobalSize(0) + tX;
        iters[tIndex] = count;
        lastValuesR[tIndex] = zr.doubleValue();
        lastValuesI[tIndex] = zi.doubleValue();

        if (calcDistance[0]) {
            distancesR[tIndex] = dr.doubleValue();
            distancesI[tIndex] = di.doubleValue();
        }

    }

    @Override
    public String toString() {
        return "DoubleDouble";
    }
}

