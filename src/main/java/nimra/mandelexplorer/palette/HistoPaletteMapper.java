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

/**
 * Created: 27.12.19   by: Armin Haaf
 *
 * from https://en.wikipedia.org/wiki/Mandelbrot_set
 *
 * @author Armin Haaf
 */
public class HistoPaletteMapper extends GradientPaletteMapper {

    private int[] numIterations;
    private int totalIterations;
    
    @Override
    public void init(final MandelParams pMandelParams) {
        super.init(pMandelParams);

        numIterations = new int[pMandelParams.getMaxIterations()];
        totalIterations = 0;
    }

    @Override
    public void prepare(final int pIter, final double pLastR, final double pLastI, final double pDistanceR, final double pDistanceI) {
        if (pIter != mandelParams.getMaxIterations()) {
            numIterations[pIter]+=pIter;
            totalIterations+=pIter;
        }
    }

    @Override
    public int map(final int pIter, final double pLastR, final double pLastI, final double pDistanceR, final double pDistanceI) {
        if (pIter != mandelParams.getMaxIterations()) {
            int tAddedIter = 0;
            for ( int i=0; i<pIter; i++) {
                tAddedIter+=numIterations[i];
            }
            final float h = tAddedIter / (float)totalIterations;

            return getRBGColor(h);
        } else {
            return insideColor.getRGB();
        }
    }

    @Override
    public String getName() {
        return "Histo";
    }

    
}
