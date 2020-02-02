package nimra.mandelexplorer.cuda;

import java.math.BigDecimal;

/**
 * Created: 25.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class CudaFloatMandelImpl extends CudaDoubleMandelImpl {

    public CudaFloatMandelImpl() {
        super("/cuda/FloatMandel.ptx");
        setPixelPrecision(new BigDecimal("1E-7"));
    }

    @Override
    public String toString() {
        return "Cuda Float";
    }

}
