package nimra.mandelexplorer.opencl;

import java.util.Scanner;

/**
 * Created: 18.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class FloatOpenCLMandelImpl extends OpenCLMandelImpl {
    public FloatOpenCLMandelImpl() {
        super(new Scanner(OpenCLDevice.class.getResourceAsStream("/opencl/FloatMandel.cl"), "UTF-8").useDelimiter("\\A").next());
    }

    @Override
    public String toString() {
        return "Float OpenCL";
    }
}
