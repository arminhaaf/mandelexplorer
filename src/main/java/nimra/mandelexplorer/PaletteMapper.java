package nimra.mandelexplorer;

import org.beryx.awt.color.ColorFactory;
import org.json.JSONObject;

import java.awt.Color;
import java.lang.reflect.Field;

/**
 * Created: 27.12.19   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public abstract class PaletteMapper {
    protected MandelParams mandelParams;

    protected Color insideColor = Color.black;

    public void init(final MandelParams pMandelParams) {
        mandelParams = pMandelParams;
    }

    public void prepare(int pIter, final double pLastR, final double pLastI, final double pDistanceR, final double pDistanceI) {
    }

    public void startMap() {
    }

    public abstract int map(int pIter, final double pLastR, final double pLastI, final double pDistanceR, final double pDistanceI);

    public String getDescription() {
        return getName();
    }

    public JSONObject toJson() {
        final JSONObject tJSONObject = new JSONObject();
        toJson(tJSONObject);
        return tJSONObject;
    }

    protected void toJson(final JSONObject pJSONObject) {
        pJSONObject.put("insideColor", toColorString(insideColor));
    }

    public void fromJson(JSONObject pJSONObject) {
        insideColor = ColorFactory.valueOf(pJSONObject.optString("insideColor", "black").trim());
    }

    protected String toColorString(Color pColor) {
        for (Field tDeclaredField : ColorFactory.class.getDeclaredFields()) {
            if (Color.class.equals(tDeclaredField.getType())) {
                try {
                    Color tColor = (Color)tDeclaredField.get(null);
                    if (pColor.equals(tColor)) {
                        return tDeclaredField.getName().toLowerCase();
                    }
                } catch (IllegalAccessException pE) {
                    pE.printStackTrace();
                }
            }
        }
        return "#" + String.format("%06X", pColor.getRGB() & 0xffffff);
    }

    public abstract String getName();

    @Override
    public String toString() {
        return getName();
    }
}
