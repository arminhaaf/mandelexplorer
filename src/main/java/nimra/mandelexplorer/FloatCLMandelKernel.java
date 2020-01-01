package nimra.mandelexplorer;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.OpenCLDevice;
import com.aparapi.internal.kernel.KernelManager;

/**
 * Created: 31.12.19   by: Armin Haaf
 * <p>
 * A DoubleDouble implementation
 *
 * @author Armin Haaf
 */
public class FloatCLMandelKernel extends MandelKernel {


    private static final FloatCLMandel floatCLMandel;

    static {
        FloatCLMandel tImpl = null;
        try {
            tImpl = ((OpenCLDevice)KernelManager.instance().bestDevice()).bind(FloatCLMandel.class);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        floatCLMandel = tImpl;
    }

    /**
     * Maximum iterations we will check for.
     */
    private int maxIterations = 100;

    private float xStart;
    private float yStart;

    private float xInc;
    private float yInc;

    private float escapeSqr;


    public FloatCLMandelKernel(final int pWidth, final int pHeight) {
        super(pWidth, pHeight);
    }

    @Override
    public void init(final MandelParams pMandelParams) {
        maxIterations = pMandelParams.getMaxIterations();
        escapeSqr = (float)(pMandelParams.getEscapeRadius() * pMandelParams.getEscapeRadius());

        double tScaleX = pMandelParams.getScale() * (width / (double)height);
        double tScaleY = pMandelParams.getScale();
        xStart = (float)(pMandelParams.getX() - tScaleX / 2.0);
        yStart = (float)(pMandelParams.getY() - tScaleY / 2.0);
        xInc = (float)(tScaleX / (double)width);
        yInc = (float)(tScaleY / (double)height);
    }


    @Override
    public synchronized Kernel execute(Range pRange) {
        if ( floatCLMandel==null ) {
            throw new RuntimeException("need open cl for " + this);
        }
        floatCLMandel.computeMandelBrot(pRange, iters, xStart, yStart,
                                        xInc, yInc, maxIterations, escapeSqr);

        return this;
    }


    @Override
    public void run() {
        throw new RuntimeException("not used");
    }

    @Override
    public String toString() {
        return "FloatCL";
    }
}

