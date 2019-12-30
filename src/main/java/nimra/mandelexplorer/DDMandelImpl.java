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

    private boolean calcDistance = false;

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
    }

    public void run() {
        final int tX = getGlobalId(0);
        final int tY = getGlobalId(1);


        /** Translate the gid into an x an y value. */
        final DD x = scaleX.multiply(tX).subtract(scaleX.divide(2).multiply(width)).divide(width).add(offsetx);
        final DD y = scaleY.multiply(tY).subtract(scaleY.divide(2).multiply(height)).divide(height).add(offsety);

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

            if (calcDistance) {
                new_dr = zr.multiply(dr).subtract(zi.multiply(di)).multiply(2.0).add(1.0);
                di = zr.multiply(di).add(zi.multiply(dr)).multiply(2.0);
                dr = new_dr;
            }

            new_zr = zrsqr.subtract(zisqr).add(x);
            zi = zr.multiply(zi).multiply(2.0).add(y);
            zr = new_zr;

            //If in a periodic orbit, assume it is trapped
            if (zr.isZero() && zi.isZero()) {
                count = maxIterations;
            } else {
                zrsqr = zr.sqr();
                zisqr = zi.sqr();
                count++;
            }
        }

        final int tIndex = tY * getGlobalSize(0) + tX;
        iters[tIndex] = count;
        lastValuesR[tIndex] = zr.doubleValue();
        lastValuesI[tIndex] = zi.doubleValue();

        if (calcDistance) {
            distancesR[tIndex] = dr.doubleValue();
            distancesI[tIndex] = di.doubleValue();
        }

    }

    @Override
    public String toString() {
        return "DoubleDouble";
    }
}

