package nimra.mandelexplorer.opencl;

import java.math.BigDecimal;
import java.util.Scanner;

/**
 * Created: 18.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class FloatOpenCLMandelImpl extends OpenCLMandelImpl {
    public FloatOpenCLMandelImpl() {
        super(new Scanner(OpenCLDevice.class.getResourceAsStream("/opencl/FloatMandel.cl"), "UTF-8").useDelimiter("\\A").next());

        setPixelPrecision(new BigDecimal("1E-7"));
    }

    @Override
    public String toString() {
        return "Float OpenCL";
    }
}
