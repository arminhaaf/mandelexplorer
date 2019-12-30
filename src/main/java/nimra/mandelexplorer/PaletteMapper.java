/**
 * Copyright (c) 2016 - 2018 Syncleus, Inc.
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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
