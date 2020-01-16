package nimra.mandelexplorer;

import nimra.mandelexplorer.util.ColorUtils;
import org.beryx.awt.color.ColorFactory;
import org.json.JSONObject;

import java.awt.Color;

/**
 * Created: 27.12.19   by: Armin Haaf
 *
 * Map compute results to colors. Palette mapping works in 4 steps:
 * <ol>
 *     <li>init with calc parameters</li>
 *     <li>prepare every pixel</li>
 *     <li>start mapping</li>
 *     <li>map every pixel</li>
 * </ol>
 *
 * Palettemappers need to be Cloneable -> so they don't need to be thread safe
 *
 * @author Armin Haaf
 */
public abstract class PaletteMapper implements Cloneable {
    protected MandelParams mandelParams;

    protected Color insideColor = Color.black;

    public void init(final MandelParams pMandelParams) {
        mandelParams = pMandelParams;
    }

    public void prepare(int pIter, final double pLastR, final double pLastI, final double pDistanceR, final double pDistanceI) {
    }

    public void startMap() {
    }

    public boolean supportsMode(CalcMode pMode) {
        return true;
    }

    public abstract int map(int pIter, final double pLastR, final double pLastI, final double pDistanceR, final double pDistanceI);

    protected int getMaxIterations() {
        return mandelParams.getMaxIterations();
    }

    public String getDescription() {
        return getName();
    }

    public JSONObject toJson() {
        final JSONObject tJSONObject = new JSONObject();
        toJson(tJSONObject);
        return tJSONObject;
    }

    protected void toJson(final JSONObject pJSONObject) {
        pJSONObject.put("insideColor", ColorUtils.toColorString(insideColor));
    }

    public void fromJson(JSONObject pJSONObject) {
        insideColor = ColorFactory.valueOf(pJSONObject.optString("insideColor", "black").trim());
    }

    public abstract String getName();

    @Override
    public String toString() {
        return getName();
    }

    @Override
    public PaletteMapper clone() {
        try {
            return (PaletteMapper)super.clone();
        } catch (CloneNotSupportedException pE) {
            throw new RuntimeException(pE);
        }
    }
}
