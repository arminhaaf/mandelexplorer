package nimra.mandelexplorer.palette;

import nimra.mandelexplorer.PaletteMapper;
import org.json.JSONObject;

import java.awt.Color;

/**
 * Created: 27.12.19   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class HuePaletteMapper extends PaletteMapper {

    protected float saturation = 1f;
    protected float brightness = 1f;

    @Override
    public int map(final int pIter, final double pLastR, final double pLastI, final double pDistanceR, final double pDistanceI) {
        if (pIter != mandelParams.getMaxIterations()) {
            final float h = (float)(pIter) / (float)mandelParams.getMaxIterations();
            return Color.HSBtoRGB(h, saturation, brightness);
        } else {
            return insideColor.getRGB();
        }
    }

    @Override
    public String getName() {
        return "Hue";
    }


    @Override
    protected void toJson(final JSONObject pJSONObject) {
        super.toJson(pJSONObject);
        pJSONObject.put("saturation", saturation);
        pJSONObject.put("brightness", brightness);
    }

    @Override
    public void fromJson(final JSONObject pJSONObject) {
        super.fromJson(pJSONObject);
        saturation = pJSONObject.optFloat("saturation", 1);
        brightness = pJSONObject.optFloat("brightness", 1);
    }

}
