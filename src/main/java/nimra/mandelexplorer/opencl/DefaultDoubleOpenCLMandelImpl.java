package nimra.mandelexplorer.opencl;

import java.util.Scanner;

/**
 * Created: 18.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class DefaultDoubleOpenCLMandelImpl extends OpenCLMandelImpl {
    public DefaultDoubleOpenCLMandelImpl() {
        super(new Scanner(OpenCLDevice.class.getResourceAsStream("/opencl/DoubleMandel.cl"), "UTF-8").useDelimiter("\\A").next());
    }

    @Override
    public String toString() {
        return "Double OpenCL";
    }
}