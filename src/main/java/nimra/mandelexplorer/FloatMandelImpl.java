package nimra.mandelexplorer;

/**
 * Created: 26.12.19   by: Armin Haaf
 * <p>
 * Implementation using doubles
 *
 * @author Armin Haaf
 */
public class FloatMandelImpl extends MandelKernel {

    /**
     * Maximum iterations we will check for.
     */
    private int maxIterations = 100;

    private float xStart;
    private float yStart;

    private float xInc;
    private float yInc;

    private float escapeSqr;


    public FloatMandelImpl(final int pWidth, final int pHeight) {
        super(pWidth, pHeight);
    }

    @Override
    public void init(final MandelParams pMandelParams) {
        maxIterations = pMandelParams.getMaxIterations();
        escapeSqr = (float) (pMandelParams.getEscapeRadius() * pMandelParams.getEscapeRadius());

        double tScaleX = pMandelParams.getScale() * (width / (double) height);
        double tScaleY = pMandelParams.getScale();
        xStart =  (float)(pMandelParams.getX() - tScaleX / 2.0);
        yStart =  (float)(pMandelParams.getY() - tScaleY / 2.0);
        xInc = (float)(tScaleX/(double)width);
        yInc = (float)(tScaleY/(double)height);

        final float escape = escapeSqr;

    }

    public void run() {
        final int tX = getGlobalId(0);
        final int tY = getGlobalId(1);


        /** Translate the gid into an x an y value. */
        final float x = xStart + tX*xInc;
        final float y = yStart + tY*yInc;

        int count = 0;

        float zr = x;
        float zi = y;
        float new_zr;

//        // Iterate until the algorithm converges or until maxIterations are reached.
//        while ((count < maxIterations) && (((zx * zx) + (zy * zy)) < 8)) {
//            new_zx = ((zx * zx) - (zy * zy)) + x;
//            zy = (2 * zx * zy) + y;
//            zx = new_zx;
//            count++;
//        }
//

        // cache the squares -> 10% faster
        float zrsqr = zr * zr;
        float zisqr = zi * zi;

        // distance
        float dr = 1;
        float di = 0;
        float new_dr;

        while ((count < maxIterations) && ((zrsqr + zisqr) < escapeSqr)) {

            if ( calcDistance[0]) {
                new_dr = 2.0f * (zr * dr - zi * di) + 1.0f;
                di = 2.0f * (zr * di + zi * dr);
                dr = new_dr;
            }

            new_zr = (zrsqr - zisqr) + x;
            zi = (2 * zr * zi) + y;
            zr = new_zr;

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
        return "Float";
    }
}
