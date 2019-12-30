/**
 * Copyright (c) 2016 - 2018 Syncleus, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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
        if (pIter != mandelParams.getMaxIterations()) {

            final float h = (float)(pIter) / (float)mandelParams.getMaxIterations();
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
