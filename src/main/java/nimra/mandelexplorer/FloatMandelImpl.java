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

    /**
     * Mutable values of scale, offsetx and offsety so that we can modify the zoom level and position of a view.
     */
    private float scaleX = .0f;
    private float scaleY = .0f;

    private float offsetx = .0f;

    private float offsety = .0f;

    private float escapeSqr = 4;


    public FloatMandelImpl(final int pWidth, final int pHeight) {
        super(pWidth, pHeight);
    }

    @Override
    public void init(final MandelParams pMandelParams) {
        scaleX = (float)pMandelParams.getScale() * width / (float)height;
        scaleY = (float)pMandelParams.getScale();
        offsetx = (float)pMandelParams.getX();
        offsety = (float)pMandelParams.getY();
        maxIterations = pMandelParams.getMaxIterations();
        escapeSqr = (float)(pMandelParams.getEscapeRadius() * pMandelParams.getEscapeRadius());
    }

    public void run() {
        final int tX = getGlobalId(0);
        final int tY = getGlobalId(1);


        /** Translate the gid into an x an y value. */
        final float tScaledWidth = (scaleX / 2) * width;
        final float x = (((tX * scaleX) - tScaledWidth) / width) + offsetx;

        final float tScaledHeight = (scaleY / 2) * height;
        final float y = (((tY * scaleY) - tScaledHeight) / height) + offsety;

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
