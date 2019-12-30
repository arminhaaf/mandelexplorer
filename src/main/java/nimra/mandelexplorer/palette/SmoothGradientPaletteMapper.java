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
package nimra.mandelexplorer.palette;

/**
 * Created: 27.12.19   by: Armin Haaf
 *
 * https://stackoverflow.com/questions/369438/smooth-spectrum-for-mandelbrot-set-rendering
 *
 * @author Armin Haaf
 */
public class SmoothGradientPaletteMapper extends GradientPaletteMapper {

    @Override
    public int map(final int pIter, final double pLastR, final double pLastI, final double pDistanceR, final double pDistanceI) {
        if (pIter != mandelParams.getMaxIterations()) {
            double log_zn = Math.log(pLastR * pLastR + pLastI * pLastI) / 2.0;
            double nu = Math.log(log_zn / Math.log(mandelParams.getEscapeRadius())) / Math.log(2.0);
            final float h = (float)(((double)pIter + 1.0 - nu) / (double)mandelParams.getMaxIterations());
            return getRBGColor(0.95f + 10 * h);
        } else {
            return insideColor.getRGB();
        }
    }

    @Override
    public String getName() {
        return "SmoothGradient";
    }
}
