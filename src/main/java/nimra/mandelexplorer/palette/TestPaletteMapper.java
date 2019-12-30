package nimra.mandelexplorer.palette;

import nimra.mandelexplorer.MandelParams;
import nimra.mandelexplorer.PaletteMapper;

import java.awt.Color;

/**
 * Created: 27.12.19   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class TestPaletteMapper extends PaletteMapper {

    int[] histo;

    @Override
    public void init(final MandelParams pMandelParams) {
        super.init(pMandelParams);

        histo = new int[pMandelParams.getMaxIterations()];
    }

    @Override
    public void prepare(final int pIter, final double pLastR, final double pLastI, final double pDistanceR, final double pDistanceI) {
        if (pIter != mandelParams.getMaxIterations()) {
            histo[pIter]++;
        }
    }

    @Override
    public int map(final int pIter, final double pLastR, final double pLastI, final double pDistanceR, final double pDistanceI) {
        if (pIter != mandelParams.getMaxIterations()) {
            double tPow = Math.pow(2, pIter);
            double tR = pLastR*pLastR+pLastI*pLastI;
            double tValue = (Math.log(tR) / tPow);

            System.out.println(tValue);

            return Color.HSBtoRGB(1f, 1f, (float)tValue);
        } else {
            return insideColor.getRGB();
        }
    }

    @Override
    public String getName() {
        return "Test";
    }

    public static void main(String[] args) {
        int maxIter = 10;
        for (int i = 1; i <= maxIter; i++) {
            final float tV = i / (float)maxIter;
            
            System.out.println(i + " -> " + (Math.log(tV)));
        }
    }


}
