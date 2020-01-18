package nimra.mandelexplorer.old;

import com.aparapi.Range;
import com.aparapi.opencl.OpenCL;

@OpenCL.Resource("FFMandel.cl")
public interface FFCLMandel extends OpenCL<FFCLMandel> {

        FFCLMandel computeMandelBrot(//
                                     Range range, //
                                     @GlobalWriteOnly("iters") int[] iters, //
                                     @GlobalWriteOnly("iters")  double[] lastValuesR,
                                     @GlobalWriteOnly("iters")  double[] lastValuesI,
                                     @GlobalWriteOnly("iters")  double[] distancesR,
                                     @GlobalWriteOnly("iters")  double[] distancesI,
                                     @Arg("calcDistance") int calcDistance,
                                     @GlobalReadOnly("startX") float[] startX, //
                                     @GlobalReadOnly("startY") float[] startY, //
                                     @GlobalReadOnly("incX") float[] incX,
                                     @GlobalReadOnly("incY") float[] incY,
                                     @Arg("maxIterations") int maxIterations,
                                     @Arg("sqrEscapeRadius") double sqrEscapeRadius
        );

}
