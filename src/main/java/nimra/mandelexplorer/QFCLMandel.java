package nimra.mandelexplorer;

import com.aparapi.Range;
import com.aparapi.opencl.OpenCL;

@OpenCL.Resource("QFMandel.cl")
public interface QFCLMandel extends OpenCL<QFCLMandel> {


    QFCLMandel computeMandelBrot(//
                                 Range range, //
                                 @GlobalWriteOnly("iters") int[] iters, //
                                 @Arg("startX") double startX, //
                                 @Arg("startY") double startY,//
                                 @Arg("incX") double incX,
                                 @Arg("incY") double incY,
                                 @Arg("maxIterations") int maxIterations,
                                 @Arg("sqrEscapeRadius") double sqrEscapeRadius
    );

}
