package nimra.mandelexplorer.cuda;

/**
 * Created: 25.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class CudaFloatMandelImpl extends CudaDoubleMandelImpl {

    public CudaFloatMandelImpl() {
        super("/cuda/FloatMandel.ptx");
    }

    @Override
    public String toString() {
        return "Cuda Float";
    }

}
