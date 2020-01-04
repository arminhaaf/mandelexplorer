package nimra.mandelexplorer;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.Device;
import com.aparapi.device.OpenCLDevice;
import com.aparapi.internal.kernel.KernelManager;

import java.lang.ref.SoftReference;
import java.math.BigDecimal;
import java.util.HashMap;
import java.util.Map;
import java.util.WeakHashMap;

/**
 * Created: 31.12.19   by: Armin Haaf
 * <p>
 * A DoubleDouble implementation
 *
 * @author Armin Haaf
 */
public class DDCLMandelKernel extends DDMandelImpl {

    public DDCLMandelKernel(final int pWidth, final int pHeight) {
        super(pWidth, pHeight);
    }

    private double[] convertToDD(DD pDD) {
        double[] tDD = new double[2];

        tDD[0] = pDD.getHi();
        tDD[1] = pDD.getLo();
        return tDD;
    }

    @Override
    public synchronized Kernel execute(Range pRange) {

        DDCLMandel tImpl = CLImplCache.getImpl(this, DDCLMandel.class);
        tImpl.computeMandelBrot(pRange, iters, lastValuesR, lastValuesI, distancesR, distancesI, calcDistance[0] ? 1 : 0,
                convertToDD(xStart), convertToDD(yStart), convertToDD(xInc), convertToDD(yInc), maxIterations, escapeSqr.getHi());

        return this;
    }


    @Override
    public void run() {
        throw new RuntimeException("not used");
    }

    @Override
    public String toString() {
        return "DDCL";
    }
}

