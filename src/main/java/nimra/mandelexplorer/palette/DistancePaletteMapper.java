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

import nimra.mandelexplorer.MandelParams;

import java.awt.Color;
import java.util.Properties;

/**
 * Created: 27.12.19   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class DistancePaletteMapper extends GradientPaletteMapper {

    protected double maxDistance;

    public DistancePaletteMapper() {
        insideColor = new Color(0x2c0091);
    }

    @Override
    public void init(final MandelParams pMandelParams) {
        super.init(pMandelParams);
        maxDistance = 0;
    }

    @Override
    public void prepare(final int pIter, final double pLastR, final double pLastI, final double pDistanceR, final double pDistanceI) {
        final double tDistance = calcDistance(pLastR, pLastI, pDistanceR, pDistanceI);
        if (Double.isFinite(tDistance)) {
            maxDistance = Math.max(tDistance, maxDistance);
        }
    }

    @Override
    public int map(final int pIter, final double pLastR, final double pLastI, final double pDistanceR, final double pDistanceI) {
        if (pIter != mandelParams.getMaxIterations()) {
            final double tDistance = calcDistance(pLastR, pLastI, pDistanceR, pDistanceI);;
            if (!Double.isFinite(tDistance)) {
                return insideColor.getRGB();
            }
//
            //final float h = (float)(1.0 - pDistance / maxDistance);
            final float h = (float)(1.0 - Math.sin(Math.PI / 2 * tDistance / maxDistance));
            // double to float conversion
            if (h == 1.0f) {
                return insideColor.getRGB();
            }
            return getRBGColor(h);
        } else {
            return insideColor.getRGB();
        }
    }

    protected double calcDistance(final double pLastR, final double pLastI, final double dr, final double di) {
        final double absZ2 = pLastR * pLastR + pLastI * pLastI;
        final double absdZ2 = dr * dr + di * di;
        return Math.sqrt(absZ2 / absdZ2) * Math.log(absZ2);
    }

    @Override
    public String getName() {
        return "Distance";
    }

}
