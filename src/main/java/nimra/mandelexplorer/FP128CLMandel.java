package nimra.mandelexplorer;

import com.aparapi.Range;
import com.aparapi.opencl.OpenCL;

@OpenCL.Resource("FP128Mandel.cl")
public interface FP128CLMandel extends OpenCL<FP128CLMandel> {

    FP128CLMandel computeMandelBrot(//
            Range range, //
            @GlobalWriteOnly("iters") int[] iters, //
            @GlobalWriteOnly("iters") double[] lastValuesR,
            @GlobalWriteOnly("iters") double[] lastValuesI,
            @GlobalWriteOnly("iters") double[] distancesR,
            @GlobalWriteOnly("iters") double[] distancesI,
            @Arg("calcDistance") int calcDistance,
            @GlobalReadOnly("startX") int[] startX, //
            @GlobalReadOnly("startY") int[] startY, //
            @GlobalReadOnly("incX") int[] incX,
            @GlobalReadOnly("incY") int[] incY,
            @Arg("maxIterations") int maxIterations,
            @Arg("sqrEscapeRadius") int sqrEscapeRadius
    );

}
