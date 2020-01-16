package nimra.mandelexplorer.palette;

import nimra.mandelexplorer.CalcMode;
import nimra.mandelexplorer.MandelParams;
import org.json.JSONObject;

/**
 * Created: 27.12.19   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class DistancePaletteMapper extends GradientPaletteMapper {

    protected double maxDistance;

    public DistancePaletteMapper() {
    }

    @Override
    protected void initDefaults() {
        fromJson(new JSONObject("{\n" +
                                "  \"insideColor\": \"black\",\n" +
                                "  \"gradients\": [\n" +
                                "    {\n" +
                                "      \"slicePercent\": 80,\n" +
                                "      \"toColor\": \"orange\",\n" +
                                "      \"fromColor\": \"red\"\n" +
                                "    },\n" +
                                "    {\n" +
                                "      \"slicePercent\": 10,\n" +
                                "      \"toColor\": \"yellow\",\n" +
                                "      \"fromColor\": \"orange\"\n" +
                                "    },\n" +
                                "    {\n" +
                                "      \"slicePercent\": 10,\n" +
                                "      \"toColor\": \"blue\",\n" +
                                "      \"fromColor\": \"yellow\"\n" +
                                "    }\n" +
                                "  ]\n" +
                                "}"));
    }

    @Override
    public boolean supportsMode(final CalcMode pMode) {
        return pMode == CalcMode.MANDELBROT_DISTANCE;
    }

    @Override
    public void init(final MandelParams pMandelParams) {
        super.init(pMandelParams);
        maxDistance = 0;
    }

    @Override
    public void prepare(final int pIter, final double pLastR, final double pLastI, final double pDistanceR, final double pDistanceI) {
        final double tDistance = calcDistance(pLastR, pLastI, pDistanceR, pDistanceI);
        if (Double.isFinite(tDistance)) {
            maxDistance = Math.max(tDistance, maxDistance);
        }
    }

    @Override
    public int map(final int pIter, final double pLastR, final double pLastI, final double pDistanceR, final double pDistanceI) {
        if (pIter != getMaxIterations()) {
            final double tDistance = calcDistance(pLastR, pLastI, pDistanceR, pDistanceI);
            if (!Double.isFinite(tDistance)) {
                return insideColor.getRGB();
            }
//
            //final float h = (float)(1.0 - pDistance / maxDistance);
            final float h = (float)(1.0 - Math.sin(Math.PI / 2 * tDistance / maxDistance));
            // double to float conversion
            if (h == 1.0f) {
                return insideColor.getRGB();
            }
            return getRBGColor(h);
        } else {
            return insideColor.getRGB();
        }
    }

    protected double calcDistance(final double pLastR, final double pLastI, final double dr, final double di) {
        final double absZ2 = pLastR * pLastR + pLastI * pLastI;
        final double absdZ2 = dr * dr + di * di;
        return Math.sqrt(absZ2 / absdZ2) * Math.log(absZ2);
    }

    @Override
    public String getName() {
        return "Distance (Escape-Radius>100)";
    }

}
