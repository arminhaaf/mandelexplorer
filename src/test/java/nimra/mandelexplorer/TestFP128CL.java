package nimra.mandelexplorer;

import com.aparapi.Range;

import java.math.BigDecimal;

/**
 * Created: 03.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class TestFP128CL {

    public static void main(String[] args) {
        FP128CLMandelKernel tFP128CLMandelKernel = new FP128CLMandelKernel(1024, 1024);

        Range range = Range.create2D(1, 1);
        final MandelParams tMandelParams = new MandelParams();
        tFP128CLMandelKernel.init(tMandelParams);
        tFP128CLMandelKernel.execute(range);

        final int[] tInts = tFP128CLMandelKernel.iters;
        final double[] tLastR = tFP128CLMandelKernel.lastValuesR;

        String[] tInfos = {
                "leftX=" + tFP128CLMandelKernel.getxStart(),
                "topY=" + tFP128CLMandelKernel.getyStart(),
                "stepX",
                "stepY",
                "add128(leftX, stepX)=" + tFP128CLMandelKernel.getxStart().add(tFP128CLMandelKernel.getxInc()),
                "add128(stepX, stepX)=" + tFP128CLMandelKernel.getxInc().add(tFP128CLMandelKernel.getxInc()),
                "mulfpu(stepX, stepY)=" + tFP128CLMandelKernel.getxInc().multiply(tFP128CLMandelKernel.getyInc()),
                "mulfp(topY, stepY)=" + tFP128CLMandelKernel.getyStart().multiply(tFP128CLMandelKernel.getyInc()),
                "set128(X)",
                "set128(Y)",
                "xc",
                "yc",
                "mul128(stepX, 14)=" + tFP128CLMandelKernel.getxInc().multiply(new BigDecimal(14)),
                "mul128(stepY, 19)=" + tFP128CLMandelKernel.getyInc().multiply(new BigDecimal(-19)),
                "(int4)(set128(1) < set128(0)",
                "(int4)(set128(-1) < set128(0)",
                "(int4)(set128(0) < set128(1)",
                "(int4)(set128(0) < set128(-1)",

                };


        for (int i = 0; i < tInfos.length; i++) {
            final FP128 tFP128 = new FP128(tInts, i);
            System.out.println(tInfos[i] + " : " + tFP128 + "   " + tFP128.toHexString());
        }

        System.out.println("\n\n\nconvert to double");

        for (int i = 0; i < tInfos.length; i++) {
            System.out.println(tInfos[i] + " : " + tLastR[i]);
        }


    }
}
