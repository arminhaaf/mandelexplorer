package nimra.mandelexplorer.math;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import java.math.RoundingMode;

/**
 * Created: 03.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class FP128 {
    private static final BigDecimal BD_2_HOCH_32 = new BigDecimal(new BigInteger("4294967296"));

    public final int[] vec;

    public FP128() {
        vec = new int[4];
    }

    public FP128(final FP128 other) {
        vec = new int[] {other.vec[0], other.vec[1], other.vec[2], other.vec[3]};
    }

    public FP128(final int pX, final int pY, final int pZ, final int pW) {
        vec = new int[]{pX, pY, pZ, pW};
    }

    public FP128(int[] pInts, int pStartIndex) {
        vec = new int[]{pInts[pStartIndex * 4 + 0], pInts[pStartIndex * 4 + 1], pInts[pStartIndex * 4 + 2], pInts[pStartIndex * 4 + 3]};
    }

    public static FP128 from(int pValue) {
        final FP128 tFP128 = new FP128(Math.abs(pValue), 0, 0, 0);
        if (pValue < 0) {
            tFP128.neg();
        }
        return tFP128;
    }

    public void neg() {
        for (int i = 0; i < vec.length; i++) {
            vec[i] ^= 0xFFFFFFFF;
        }
        bitInc();
    }

    public void bitInc() {
        int[] h = new int[4];
        for (int i = 0; i < vec.length; i++) {
            h[i] = vec[i] == 0xFFFFFFFF ? 1 : 0;
        }
        int[] c = new int[]{
                h[1] & h[2] & h[3] & 1,
                h[2] & h[3] & 1,
                h[3] & 1,
                1
        };

        for (int i = 0; i < vec.length; i++) {
            vec[i] = vec[i] + c[i];
        }
    }

    public void copy(int[] pData, int tIndex) {
        System.arraycopy(vec, 0, pData, tIndex * 4, vec.length);
    }

    public static FP128 from(final BigDecimal pValue) {
        final boolean tNeg = pValue.compareTo(BigDecimal.ZERO) < 0;

        BigDecimal tValue = pValue.abs();

        final FP128 tResult = new FP128();

        final int[] tInts = tResult.vec;


        for (int i = 0; i < tInts.length; i++) {
            BigDecimal tIntFloor = tValue.setScale(0, RoundingMode.FLOOR);
            tValue = tValue.subtract(tIntFloor).multiply(BD_2_HOCH_32);
            tInts[i] = tIntFloor.intValue();
        }

        if (tNeg) {
            tResult.neg();
        }

        return tResult;
    }


    public String toString() {
        FP128 tCopy = new FP128(this);
        int[] tVec = tCopy.vec;
        boolean tNeg = tVec[0] < 0;

        if (tNeg) {
            tCopy.neg();
        }

        BigDecimal tBigDecimal = new BigDecimal(tVec[0]);
        for (int i = 1; i < 4; i++) {
            BigDecimal tAddNK = new BigDecimal(tVec[i]);
            for (int j = 0; j < i; j++) {
                tAddNK = tAddNK.divide(BD_2_HOCH_32, MathContext.DECIMAL128);
            }
            tBigDecimal = tBigDecimal.add(tAddNK);
        }

        // 27 decimals
        tBigDecimal = tBigDecimal.setScale(27, RoundingMode.FLOOR);
        if (tNeg) {
            return tBigDecimal.negate().toString();
        } else {
            return tBigDecimal.toString();
        }
    }

    public String toHexString() {
        StringBuilder tStringBuilder = new StringBuilder();
        for (int tX : vec) {
            tStringBuilder.append(String.format("0x%08x,", tX));
        }
        return tStringBuilder.toString();
    }

}
