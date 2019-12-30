package nimra.mandelexplorer.palette;

import org.json.JSONObject;

import java.awt.Color;

/**
 * Created: 27.12.19   by: Armin Haaf
 *
 * https://stackoverflow.com/questions/369438/smooth-spectrum-for-mandelbrot-set-rendering
 *
 * @author Armin Haaf
 */
public class SmoothHuePaletteMapper extends HuePaletteMapper {

    protected float hueOffset = 0.95f;
    protected float hueMulti = 10.0f;


    @Override
    public int map(final int pIter, final double pLastR, final double pLastI, final double pDistanceR, final double pDistanceI) {
        if (pIter != getMaxIterations()) {
            double log_zn = Math.log(pLastR * pLastR + pLastI * pLastI) / 2.0;
            double nu = Math.log(log_zn / Math.log(mandelParams.getEscapeRadius())) / Math.log(2.0);
            final float h = (float)((pIter + 1 - nu) / (double)getMaxIterations());
            return Color.HSBtoRGB(hueOffset + hueMulti * h, saturation, brightness);
        } else {
            return insideColor.getRGB();
        }
    }


    @Override
    protected void toJson(final JSONObject pJSONObject) {
        super.toJson(pJSONObject);
        pJSONObject.put("hueOffset", hueOffset);
        pJSONObject.put("hueMulti", brightness);
    }

    @Override
    public void fromJson(final JSONObject pJSONObject) {
        super.fromJson(pJSONObject);
        hueOffset = pJSONObject.optFloat("hueOffset", 1);
        brightness = pJSONObject.optFloat("hueMulti", 1);
    }


    @Override
    public String getName() {
        return "SmoothHue";
    }
}
