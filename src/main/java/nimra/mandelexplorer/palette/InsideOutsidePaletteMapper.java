package nimra.mandelexplorer.palette;

import nimra.mandelexplorer.PaletteMapper;
import nimra.mandelexplorer.util.ColorUtils;
import org.beryx.awt.color.ColorFactory;
import org.json.JSONObject;

import java.awt.Color;

/**
 * Created: 27.12.19   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class InsideOutsidePaletteMapper extends PaletteMapper {

    private Color insideColor = Color.white;
    private Color outsideColor = Color.black;

    @Override
    public int map(final int pIter, final double pLastR, final double pLastI, final double pDistanceR, final double pDistanceI) {
        if (pIter != getMaxIterations()) {
            return outsideColor.getRGB();
        } else {
            return insideColor.getRGB();
        }
    }

    @Override
    public String getName() {
        return "In&Out";
    }

    @Override
    public String getDescription() {
        return "Inside Outside Coloring";
    }


    @Override
    protected void toJson(final JSONObject pJSONObject) {
        super.toJson(pJSONObject);
        pJSONObject.put("insideColor", ColorUtils.toColorString(insideColor));
        pJSONObject.put("outsideColor", ColorUtils.toColorString(outsideColor));
    }

    @Override
    public void fromJson(final JSONObject pJSONObject) {
        super.fromJson(pJSONObject);
        insideColor = ColorFactory.valueOf(pJSONObject.optString("insideColor", "white").trim());
        outsideColor = ColorFactory.valueOf(pJSONObject.optString("outsideColor", "white").trim());
    }

}
