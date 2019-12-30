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
