package nimra.mandelexplorer.palette;

import nimra.mandelexplorer.PaletteMapper;
import org.json.JSONObject;

import java.awt.Color;

/**
 * Created: 27.12.19   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class BrightnessPaletteMapper extends PaletteMapper {
    protected float saturation = 0f;
    protected float hue = 0f;


    @Override
    public int map(final int pIter, final double pLastR, final double pLastI, final double pDistanceR, final double pDistanceI) {
        if (pIter != getMaxIterations()) {

            final float h = (float)(pIter) / (float)getMaxIterations();
            return Color.HSBtoRGB(hue, saturation, 1-(float)Math.sqrt(h));
        } else {
            return insideColor.getRGB();
        }
    }

    @Override
    public String getName() {
        return "Brightness";
    }

    @Override
    protected void toJson(final JSONObject pJSONObject) {
        super.toJson(pJSONObject);
        pJSONObject.put("saturation", saturation);
        pJSONObject.put("hue", hue);
    }

    @Override
    public void fromJson(final JSONObject pJSONObject) {
        super.fromJson(pJSONObject);
        saturation = pJSONObject.optFloat("saturation", 1);
        hue = pJSONObject.optFloat("hue", 1);
    }


}
