package nimra.mandelexplorer;

import com.aparapi.Kernel;

import java.util.ArrayList;
import java.util.List;

/**
 * Created: 26.12.19   by: Armin Haaf
 *
 * A simple abstract class for different algoritms -> it cold be shared a lot more code, however aparapi is restricted
 *
 * @author Armin Haaf
 */
public abstract class MandelKernel extends Kernel {

    // dispose the kernels at the end
    private static final List<Kernel> KERNELS = new ArrayList<>();

    protected final int width;
    protected final int height;

    /**
     * buffer used to store the iterations (width * height).
     */
    final protected int iters[];

    /**
     * buffer used to store the last calculated real value of a point -> used for some palette calculations
     */
    final protected double lastValuesR[];

    /**
     * buffer used to store the last calculated imaginary value of a point -> used for some palette calculations
     */
    final protected double lastValuesI[];

    final protected double distancesR[];
    final protected double distancesI[];


    public MandelKernel(int pWidth, int pHeight) {
        width = pWidth;
        height = pHeight;
        iters = new int[pWidth * pHeight];
        lastValuesR = new double[pWidth * pHeight];
        lastValuesI = new double[pWidth * pHeight];
        distancesR = new double[pWidth * pHeight];
        distancesI = new double[pWidth * pHeight];

        KERNELS.add(this);
    }


    // prepare the calculation
    public abstract void init(final MandelParams pMandelParams);

    public int[] getIters() {
        return iters;
    }

    public double[] getLastValuesR() {
        return lastValuesR;
    }

    public double[] getLastValuesI() {
        return lastValuesI;
    }

    public double[] getDistancesR() {
        return distancesR;
    }

    public double[] getDistancesI() {
        return distancesI;
    }


    public static void disposeAll() {
        for (Kernel tKernel : KERNELS) {
            tKernel.dispose();
        }
        KERNELS.clear();
    }
}
