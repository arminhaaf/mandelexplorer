package nimra.mandelexplorer;

/**
 * Created: 26.12.19   by: Armin Haaf
 * <p>
 * Implementation using doubles
 *
 * @author Armin Haaf
 */
public class DoubleMandelImpl extends MandelKernel {

    public int maxIterations = 100;

    protected double xStart;
    protected double yStart;

    protected double xInc;
    protected double yInc;

    protected double escapeSqr;


    public DoubleMandelImpl(final int pWidth, final int pHeight) {
        super(pWidth, pHeight);
    }

    @Override
    public void init(final MandelParams pMandelParams) {
        maxIterations = pMandelParams.getMaxIterations();
        escapeSqr = (double) (pMandelParams.getEscapeRadius() * pMandelParams.getEscapeRadius());

        double tScaleX = pMandelParams.getScale_double() * (width / (double) height);
        double tScaleY = pMandelParams.getScale_double();
        xStart = pMandelParams.getX_Double() - tScaleX / 2.0;
        yStart = pMandelParams.getY_Double() - tScaleY / 2.0;
        xInc = tScaleX/(double)width;
        yInc = tScaleY/(double)height;
    }

    public void run() {
        final int tX = getGlobalId(0);
        final int tY = getGlobalId(1);

        final double x = xStart + tX*xInc;
        final double y = yStart + tY*yInc;

        int count = 0;

        double zr = x;
        double zi = y;
        double new_zr;

//        // Iterate until the algorithm converges or until maxIterations are reached.
//        while ((count < maxIterations) && (((zx * zx) + (zy * zy)) < 8)) {
//            new_zx = ((zx * zx) - (zy * zy)) + x;
//            zy = (2 * zx * zy) + y;
//            zx = new_zx;
//            count++;
//        }
//

        // cache the squares -> 10% faster
        double zrsqr = zr * zr;
        double zisqr = zi * zi;

        // distance
        double dr = 1;
        double di = 0;
        double new_dr;

        while ((count < maxIterations) && ((zrsqr + zisqr) < escapeSqr)) {
            if ( calcDistance[0] ) {
                new_dr = 2.0 * (zr * dr - zi * di) + 1.0;
                di = 2.0 * (zr * di + zi * dr);
                dr = new_dr;
            }

            zi = (2 * zr * zi) + y;
            zr = (zrsqr - zisqr) + x;

            //If in a periodic orbit, assume it is trapped
            if (zr == 0.0 && zi == 0.0) {
                count = maxIterations;
            } else {
                zrsqr = zr * zr;
                zisqr = zi * zi;
                count++;
            }
        }

        final int tIndex = tY * getGlobalSize(0) + tX;
        iters[tIndex] = count;
        lastValuesR[tIndex] = zr;
        lastValuesI[tIndex] = zi;
        if ( calcDistance[0] ) {
            distancesR[tIndex] = dr;
            distancesI[tIndex] = di;
        }

    }

    @Override
    public String toString() {
        return "Double";
    }
}
