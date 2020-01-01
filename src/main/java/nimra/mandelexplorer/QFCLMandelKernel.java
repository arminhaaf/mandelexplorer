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
public class QFCLMandelKernel extends MandelKernel {

    private static final QFCLMandel qfCLMandel;

    static {
        QFCLMandel tImpl = null;
        try {
            tImpl = ((OpenCLDevice)KernelManager.instance().bestDevice()).bind(QFCLMandel.class);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        qfCLMandel = tImpl;
    }

    /**
     * Maximum iterations we will check for.
     */
    private int maxIterations = 100;

    private double xStart;
    private double yStart;

    private double xInc;
    private double yInc;

    private double escapeSqr;


    public QFCLMandelKernel(final int pWidth, final int pHeight) {
        super(pWidth, pHeight);
    }

    @Override
    public void init(final MandelParams pMandelParams) {
        maxIterations = pMandelParams.getMaxIterations();
        escapeSqr = pMandelParams.getEscapeRadius() * pMandelParams.getEscapeRadius();

        double tScaleX = pMandelParams.getScale() * (width / (double) height);
        double tScaleY = pMandelParams.getScale();
        xStart = pMandelParams.getX() - (tScaleX / 2.0);
        yStart = pMandelParams.getY() - tScaleY / 2.0;
        xInc = tScaleX/(double)width;
        yInc = tScaleY/(double)height;
    }

    @Override
    public synchronized Kernel execute(Range pRange) {
        if (qfCLMandel == null) {
            throw new RuntimeException("need open cl for " + this);
        }

        qfCLMandel.computeMandelBrot(pRange, iters, xStart, yStart,
                xInc, yInc, maxIterations, escapeSqr);

        return this;
    }


    @Override
    public void run() {
        throw new RuntimeException("not used");
    }

    @Override
    public String toString() {
        return "QFCL";
    }
}

