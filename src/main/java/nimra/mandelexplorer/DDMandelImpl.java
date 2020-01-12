package nimra.mandelexplorer;

/**
 * Created: 31.12.19   by: Armin Haaf
 * <p>
 * A DoubleDouble implementation
 *
 * @author Armin Haaf
 */
public class DDMandelImpl extends MandelKernel {

    protected int maxIterations = 100;

    protected DD xStart;
    protected DD yStart;

    protected DD xInc;
    protected DD yInc;

    protected double escapeSqr;

    public DDMandelImpl(final int pWidth, final int pHeight) {
        super(pWidth, pHeight);
    }


    @Override
    public void init(final MandelParams pMandelParams) {
        maxIterations = pMandelParams.getMaxIterations();

        final DD tScaleX = new DD(pMandelParams.getScale()).multiply(width).divide(height);
        final DD tScaleY = new DD(pMandelParams.getScale());
        xStart = new DD(pMandelParams.getX()).subtract(tScaleX.divide(2.0));
        yStart = new DD(pMandelParams.getY()).subtract(tScaleY.divide(2.0));
        xInc = new DD(tScaleX).divide(width);
        yInc = new DD(tScaleY).divide(height);

        escapeSqr = pMandelParams.getEscapeRadius()*pMandelParams.getEscapeRadius();
    }

    public void run() {
        final int x = getGlobalId(0);
        final int y = getGlobalId(1);

        final DD cr = xInc.multiply(x).selfAdd(xStart);
        final DD ci = yInc.multiply(y).selfAdd(yStart);

        int count = 0;

        final DD zr = new DD(cr);
        final DD zi = new DD(ci);

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

        while ((count < maxIterations) && ((zrsqr.add(zisqr)).getHi()<escapeSqr)) {

            if (calcDistance[0]) {
                tmpDD.setValue(zr).selfMultiply(dr).selfSubtract(zi.multiply(di)).selfMultiply(2.0).selfAdd(1.0);
                di.selfMultiply(zr).selfAdd(zi.multiply(dr)).selfMultiply(2.0);
                dr.setValue(tmpDD);
            }

            tmpDD.setValue(zrsqr).selfSubtract(zisqr).selfAdd(cr);
            zi.selfMultiply(zr).selfMultiply(2.0).selfAdd(ci);
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

        final int tIndex = y * getGlobalSize(0) + x;
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

