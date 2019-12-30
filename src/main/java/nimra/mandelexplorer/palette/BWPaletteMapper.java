package nimra.mandelexplorer.palette;

import nimra.mandelexplorer.PaletteMapper;
import org.json.JSONObject;

import java.awt.Color;

/**
 * Created: 27.12.19   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class BWPaletteMapper extends PaletteMapper {

    private boolean blackInside = true;

    @Override
    public int map(final int pIter, final double pLastR, final double pLastI, final double pDistanceR, final double pDistanceI) {
        if (pIter != mandelParams.getMaxIterations()) {
            if ( pIter%2 == mandelParams.getMaxIterations() % 2) {
                return blackInside ? Color.black.getRGB() : Color.white.getRGB();
            } else {
                return blackInside ? Color.white.getRGB() : Color.black.getRGB();
            }
        } else {
            return blackInside ? Color.black.getRGB() : Color.white.getRGB();
        }
    }

    @Override
    public String getName() {
        return "Black&White";
    }

    @Override
    public String getDescription() {
        return "Black and White palette";
    }


    @Override
    protected void toJson(final JSONObject pJSONObject) {
        super.toJson(pJSONObject);
        pJSONObject.put("blackInside", blackInside);
    }

    @Override
    public void fromJson(final JSONObject pJSONObject) {
        super.fromJson(pJSONObject);
        if ( pJSONObject.has("blackInside")) {
            blackInside = pJSONObject.getBoolean("blackInside");
        }
    }

}
