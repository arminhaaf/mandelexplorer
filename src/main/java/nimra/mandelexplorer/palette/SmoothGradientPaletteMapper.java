package nimra.mandelexplorer.palette;

/**
 * Created: 27.12.19   by: Armin Haaf
 *
 * https://stackoverflow.com/questions/369438/smooth-spectrum-for-mandelbrot-set-rendering
 *
 * @author Armin Haaf
 */
public class SmoothGradientPaletteMapper extends GradientPaletteMapper {

    @Override
    public int map(final int pIter, final double pLastR, final double pLastI, final double pDistanceR, final double pDistanceI) {
        if (pIter != mandelParams.getMaxIterations()) {
            double log_zn = Math.log(pLastR * pLastR + pLastI * pLastI) / 2.0;
            double nu = Math.log(log_zn / Math.log(mandelParams.getEscapeRadius())) / Math.log(2.0);
            final float h = (float)(((double)pIter + 1.0 - nu) / (double)mandelParams.getMaxIterations());
            return getRBGColor(0.95f + 10 * h);
        } else {
            return insideColor.getRGB();
        }
    }

    @Override
    public String getName() {
        return "SmoothGradient";
    }
}
