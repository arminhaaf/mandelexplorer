package nimra.mandelexplorer;

import nimra.mandelexplorer.math.FP128;

import java.math.BigDecimal;

/**
 * Created: 04.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class TestFP128 {
    public static void main(String[] args) {
        FP128 tFP128 = FP128.from(new BigDecimal("-2.5162882527147087857847976307996050"));
        //FP128 tFP128 = FP128.from(new BigDecimal("-2.5"));
        System.out.println(tFP128 + " :  " + tFP128.toHexString());

        tFP128 = FP128.from(new BigDecimal("0.0001"));
        System.out.println(tFP128 + " :  " + tFP128.toHexString());

        tFP128 = FP128.from(new BigDecimal("2.5"));
        System.out.println(tFP128 + " :  " + tFP128.toHexString());

        BigDecimal tSQRT2 = new BigDecimal("-1.414213562373095048801688724209698078569671875376948073176");
        tFP128 = FP128.from(tSQRT2);
        //FP128 tFP128 = FP128.from(new BigDecimal("-2.5"));
        System.out.println(tFP128 + " :  " + tFP128.toHexString() + " expect: " + "16a09e667f3bcc908b2fb1366");

//        1, 0x6A09E667, 0xF3BCC908, 0xB2FB1366
    }

}
