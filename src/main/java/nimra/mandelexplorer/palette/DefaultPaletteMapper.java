package nimra.mandelexplorer.palette;

import nimra.mandelexplorer.PaletteMapper;
import org.json.JSONObject;

import java.awt.Color;

/**
 * Created: 27.12.19   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class DefaultPaletteMapper extends PaletteMapper {

    protected float saturation = 1f;

    @Override
    public int map(final int pIter, final double pLastR, final double pLastI, final double pDistanceR, final double pDistanceI) {
        if (pIter != mandelParams.getMaxIterations()) {
            final float h = (float)(pIter % 256) / 256;
            final float b = 1.0f - (h * h);
            return Color.HSBtoRGB(h, saturation, b);
        } else {
            return insideColor.getRGB();
        }
    }

    @Override
    protected void toJson(final JSONObject pJSONObject) {
        super.toJson(pJSONObject);
        pJSONObject.put("saturation", saturation);
    }

    @Override
    public void fromJson(final JSONObject pJSONObject) {
        super.fromJson(pJSONObject);
        saturation = pJSONObject.optFloat("saturation", 1);
    }

    @Override
    public String getName() {
        return "Default";
    }
}
