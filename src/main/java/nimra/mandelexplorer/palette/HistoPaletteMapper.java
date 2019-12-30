package nimra.mandelexplorer.palette;

import nimra.mandelexplorer.MandelParams;

/**
 * Created: 27.12.19   by: Armin Haaf
 *
 * from https://en.wikipedia.org/wiki/Mandelbrot_set
 *
 * @author Armin Haaf
 */
public class HistoPaletteMapper extends GradientPaletteMapper {

    private int[] numIterations;
    private int totalIterations;
    
    @Override
    public void init(final MandelParams pMandelParams) {
        super.init(pMandelParams);

        numIterations = new int[getMaxIterations()];
    }

    @Override
    public void prepare(final int pIter, final double pLastR, final double pLastI, final double pDistanceR, final double pDistanceI) {
        if (pIter != getMaxIterations()) {
            numIterations[pIter]++;
        }
    }

    @Override
    public void startMap() {
        totalIterations = 0;
        for (int tNumIteration : numIterations) {
            totalIterations += tNumIteration;
        }
    }

    @Override
    public int map(final int pIter, final double pLastR, final double pLastI, final double pDistanceR, final double pDistanceI) {
        if (pIter != getMaxIterations()) {
            float tHue = 0;
            for ( int i=0; i<pIter; i++) {
                tHue+=numIterations[i]/(float)totalIterations;
            }

            // this is not normalized (not between 0 and 1)
            return getRBGColor(tHue);
        } else {
            return insideColor.getRGB();
        }
    }

    @Override
    public String getName() {
        return "Histo";
    }

    
}
