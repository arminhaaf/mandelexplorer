package nimra.mandelexplorer;

import com.aparapi.Range;
import com.aparapi.opencl.OpenCL;

@OpenCL.Resource("FloatMandel.cl")
public interface FloatCLMandel extends OpenCL<FloatCLMandel> {


        FloatCLMandel computeMandelBrot(//
                                        Range range, //
                                        @GlobalWriteOnly("iters") int[] iters, //
                                        @Arg("startX") float startX, //
                                        @Arg("startY") float startY,//
                                        @Arg("incX") float incX,
                                        @Arg("incY") float incY,
                                        @Arg("maxIterations") int maxIterations,
                                        @Arg("sqrEscapeRadius") float sqrEscapeRadius
        );

}
