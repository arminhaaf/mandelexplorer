package nimra.mandelexplorer;

/**
 * Created: 31.12.19   by: Armin Haaf
 * <p>
 * A DoubleDouble implementation
 *
 * @author Armin Haaf
 */
public class DDMandelImpl extends MandelKernel {

    /**
     * Maximum iterations we will check for.
     */
    private int maxIterations = 100;

    /**
     * Mutable values of scale, offsetx and offsety so that we can modify the zoom level and position of a view.
     */
    private DD scaleX = null;
    private DD scaleY = null;

    private DD offsetx = null;

    private DD offsety = null;

    private DD escapeSqr = null;

    private DD scaledWidth;
    private DD scaledHeight;


    public DDMandelImpl(final int pWidth, final int pHeight) {
        super(pWidth, pHeight);
    }

    @Override
    public void init(final MandelParams pMandelParams) {
        scaleX = new DD(pMandelParams.getScale() * (width / (double)height));
        scaleY = new DD(pMandelParams.getScale());
        offsetx = new DD(pMandelParams.getX());
        offsety = new DD(pMandelParams.getY());
        maxIterations = pMandelParams.getMaxIterations();
        escapeSqr = new DD(pMandelParams.getEscapeRadius() * pMandelParams.getEscapeRadius());

        scaledWidth = scaleX.divide(2).selfMultiply(width);
        scaledHeight = scaleY.divide(2).selfMultiply(height);
    }

    public void run() {
        final int tX = getGlobalId(0);
        final int tY = getGlobalId(1);

        /** Translate the gid into an x an y value. */
        final DD x = scaleX.multiply(tX).selfSubtract(scaledWidth).selfDivide(width).selfAdd(offsetx);
        final DD y = scaleY.multiply(tY).selfSubtract(scaledHeight).selfDivide(height).selfAdd(offsety);

        int count = 0;

        DD zr = x;
        DD zi = y;
        DD new_zr;

//        // Iterate until the algorithm converges or until maxIterations are reached.
//        while ((count < maxIterations) && (((zx * zx) + (zy * zy)) < 8)) {
//            new_zx = ((zx * zx) - (zy * zy)) + x;
//            zy = (2 * zx * zy) + y;
//            zx = new_zx;
//            count++;
//        }
//

        // cache the squares -> 10% faster
        DD zrsqr = zr.sqr();
        DD zisqr = zi.sqr();

        // distance
        DD dr = new DD(1);
        DD di = new DD(0);
        DD new_dr;

        while ((count < maxIterations) && ((zrsqr.add(zisqr)).lt(escapeSqr))) {

            if (calcDistance[0]) {
                new_dr = zr.multiply(dr).selfSubtract(zi.multiply(di)).selfMultiply(2.0).selfAdd(1.0);
                di = zr.multiply(di).selfAdd(zi.multiply(dr)).selfMultiply(2.0);
                dr = new_dr;
            }

            new_zr = zrsqr.subtract(zisqr).selfAdd(x);
            zi = zi.multiply(zr).selfMultiply(2.0).selfAdd(y);
            zr = new_zr;

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

