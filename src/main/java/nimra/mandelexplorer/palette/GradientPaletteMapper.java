package nimra.mandelexplorer.palette;

import nimra.mandelexplorer.PaletteMapper;
import org.beryx.awt.color.ColorFactory;
import org.json.JSONArray;
import org.json.JSONObject;

import java.awt.Color;
import java.util.ArrayList;
import java.util.List;

/**
 * Created: 29.12.19   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public abstract class GradientPaletteMapper extends PaletteMapper {
    protected List<Gradient> gradients = new ArrayList<>();

    // expect value between 0 and 1
    protected int getRBGColor(float pValue) {
        float tRatio = Math.abs(pValue) % 1.0f;

        float tSlicePercent = 0;
        for (int i = 0; i < gradients.size(); i++) {
            Gradient tGradient = gradients.get(i);
            float tNextSlicePercent;
            if (tGradient.slicePercent > 0) {
                tNextSlicePercent = tSlicePercent + tGradient.slicePercent;
            } else {
                tNextSlicePercent = tSlicePercent + 100.0f / (float)gradients.size();
            }

            if (tRatio * 100.0f < tNextSlicePercent || i + 1 == gradients.size()) {
                tRatio = (tRatio - tSlicePercent / 100.0f) * 100f / (tNextSlicePercent - tSlicePercent);

                return tGradient.getRBGColor(Math.min(tRatio, 1));
            }

            tSlicePercent = tNextSlicePercent;
        }

        return -1;
    }

    @Override
    protected void toJson(final JSONObject pJSONObject) {
        super.toJson(pJSONObject);
        JSONArray tJsonGradients = new JSONArray();
        for (int i = 0; i < gradients.size(); i++) {
            Gradient tGradient = gradients.get(i);
            tJsonGradients.put(i, tGradient.toJson());
        }
        pJSONObject.put("gradients", tJsonGradients);
    }

    @Override
    public void fromJson(final JSONObject pJSONObject) {
        super.fromJson(pJSONObject);

        gradients.clear();
        if (pJSONObject.has("gradients")) {
            JSONArray tJsonGradients = pJSONObject.getJSONArray("gradients");
            for (Object tJsonGradient : tJsonGradients) {
                final Gradient tGradient = new Gradient();
                tGradient.fromJson((JSONObject)tJsonGradient);
                gradients.add(tGradient);
            }
        }
    }


    class Gradient {

        Color fromColor;
        Color toColor;

        int slicePercent = 100;

        GradientType type = GradientType.LINEAR;

        public Gradient() {
        }

        public int getRBGColor(float pRatio) {
            return getLinearRBGColor(pRatio);
        }

        public int getLinearRBGColor(float pRatio) {
            int red = (int)(toColor.getRed() * pRatio + fromColor.getRed() * (1 - pRatio));
            int green = (int)(toColor.getGreen() * pRatio + fromColor.getGreen() * (1 - pRatio));
            int blue = (int)(toColor.getBlue() * pRatio + fromColor.getBlue() * (1 - pRatio));
            return ((red & 0xFF) << 16) |
                   ((green & 0xFF) << 8) |
                   ((blue & 0xFF));

        }


        public JSONObject toJson() {
            final JSONObject tJSONObject = new JSONObject();
            tJSONObject.put("fromColor", toColorString(fromColor));
            tJSONObject.put("toColor", toColorString(toColor));
            if (slicePercent > 0) {
                tJSONObject.put("slicePercent", slicePercent);
            }
            return tJSONObject;
        }

        public void fromJson(final JSONObject pJSONObject) {
            fromColor = ColorFactory.valueOf(pJSONObject.optString("fromColor", "red"));
            toColor = ColorFactory.valueOf(pJSONObject.optString("toColor", "blue"));
            slicePercent = Integer.parseInt(pJSONObject.optString("slicePercent", "0"));
        }
    }

    enum GradientType {
        LINEAR;
    }


}

