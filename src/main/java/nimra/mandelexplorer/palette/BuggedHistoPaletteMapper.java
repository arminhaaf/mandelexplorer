package nimra.mandelexplorer.palette;

import nimra.mandelexplorer.MandelParams;
import org.json.JSONObject;

/**
 * Created: 30.12.19   by: Armin Haaf
 * <p>
 * A palette mapper with interesting results -> it was a bugged histo implementation
 *
 * @author Armin Haaf
 */
public class BuggedHistoPaletteMapper extends GradientPaletteMapper {

    private int[] numIterations;
    private int totalIterations;

    @Override
    protected void initDefaults() {
        fromJson(new JSONObject("{\n" +
                                "  \"insideColor\": \"black\",\n" +
                                "  \"gradients\": [\n" +
                                "    {\n" +
                                "      \"toColor\": \"yellow\",\n" +
                                "      \"fromColor\": \"red\"\n" +
                                "    },\n" +
                                "    {\n" +
                                "      \"toColor\": \"green\",\n" +
                                "      \"fromColor\": \"yellow\"\n" +
                                "    },\n" +
                                "    {\n" +
                                "      \"toColor\": \"blue\",\n" +
                                "      \"fromColor\": \"green\"\n" +
                                "    }\n" +
                                "  ]\n" +
                                "}"));
    }

    @Override
    public void init(final MandelParams pMandelParams) {
        super.init(pMandelParams);

        numIterations = new int[getMaxIterations()];
        totalIterations = 0;
    }

    @Override
    public void prepare(final int pIter, final double pLastR, final double pLastI, final double pDistanceR, final double pDistanceI) {
        if (pIter != getMaxIterations()) {
            numIterations[pIter]++;
            totalIterations++;
        }
    }

    @Override
    public int map(final int pIter, final double pLastR, final double pLastI, final double pDistanceR, final double pDistanceI) {
        if (pIter != getMaxIterations()) {
            float tHue = 0;
            for (int i = 0; i < pIter; i++) {
                tHue += numIterations[i] / (float)totalIterations;
            }

            return getRBGColor(tHue);
        } else {
            return insideColor.getRGB();
        }
    }

    @Override
    public String getName() {
        return "Bugged Histo";
    }


}
