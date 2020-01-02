package nimra.mandelexplorer;

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
                                     @Arg("startX") double startX, //
                                     @Arg("startY") double startY,//
                                     @Arg("incX") double incX,
                                     @Arg("incY") double incY,
                                     @Arg("maxIterations") int maxIterations,
                                     @Arg("sqrEscapeRadius") double sqrEscapeRadius
        );

}
