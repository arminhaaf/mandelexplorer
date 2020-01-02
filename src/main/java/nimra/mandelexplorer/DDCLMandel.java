package nimra.mandelexplorer;

import com.aparapi.Range;
import com.aparapi.opencl.OpenCL;

@OpenCL.Resource("DDMandel.cl")
public interface DDCLMandel extends OpenCL<DDCLMandel> {

        DDCLMandel computeMandelBrot(//
                                     Range range, //
                                     @GlobalWriteOnly("iters") int[] iters, //
                                     @GlobalWriteOnly("iters") double[] lastValuesR,
                                     @GlobalWriteOnly("iters") double[] lastValuesI,
                                     @GlobalWriteOnly("iters") double[] distancesR,
                                     @GlobalWriteOnly("iters") double[] distancesI,
                                     @Arg("calcDistance") int calcDistance,
                                     @GlobalReadOnly("startX") double[] startX, //
                                     @GlobalReadOnly("startY") double[] startY, //
                                     @GlobalReadOnly("incX") double[] incX,
                                     @GlobalReadOnly("incY") double[] incY,
                                     @Arg("maxIterations") int maxIterations,
                                     @Arg("sqrEscapeRadius") double sqrEscapeRadius
        );

}
