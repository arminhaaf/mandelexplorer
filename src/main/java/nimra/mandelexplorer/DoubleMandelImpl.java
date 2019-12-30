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
package nimra.mandelexplorer;

/**
 * Created: 26.12.19   by: Armin Haaf
 * <p>
 * Implementation using doubles
 *
 * @author Armin Haaf
 */
public class DoubleMandelImpl extends MandelKernel {

    /**
     * Maximum iterations we will check for.
     */
    private int maxIterations = 100;

    /**
     * Mutable values of scale, offsetx and offsety so that we can modify the zoom level and position of a view.
     */
    private double scaleX = .0f;
    private double scaleY = .0f;

    private double offsetx = .0f;

    private double offsety = .0f;

    private double escapeSqr = 4;

    public DoubleMandelImpl(final int pWidth, final int pHeight) {
        super(pWidth, pHeight);
    }

    @Override
    public void init(final MandelParams pMandelParams) {
        scaleX = pMandelParams.getScale() * (width / (double)height);
        scaleY = pMandelParams.getScale();
        offsetx = pMandelParams.getX();
        offsety = pMandelParams.getY();
        maxIterations = pMandelParams.getMaxIterations();
        escapeSqr = pMandelParams.getEscapeRadius() * pMandelParams.getEscapeRadius();
    }

    public void run() {
        final int tX = getGlobalId(0);
        final int tY = getGlobalId(1);


        /** Translate the gid into an x an y value. */
        final double x = (((tX * scaleX) - ((scaleX / 2) * width)) / width) + offsetx;

        final double y = (((tY * scaleY) - ((scaleY / 2) * height)) / height) + offsety;

        int count = 0;

        double zr = x;
        double zi = y;
        double new_zr;

//        // Iterate until the algorithm converges or until maxIterations are reached.
//        while ((count < maxIterations) && (((zx * zx) + (zy * zy)) < 8)) {
//            new_zx = ((zx * zx) - (zy * zy)) + x;
//            zy = (2 * zx * zy) + y;
//            zx = new_zx;
//            count++;
//        }
//

        // cache the squares -> 10% faster
        double zrsqr = zr * zr;
        double zisqr = zi * zi;

        // distance
        double dr = 1;
        double di = 0;
        double new_dr;

        while ((count < maxIterations) && ((zrsqr + zisqr) < escapeSqr)) {

            new_dr = 2.0 * (zr * dr - zi * di) + 1.0;
            di = 2.0 * (zr * di + zi * dr);
            dr = new_dr;

            new_zr = (zrsqr - zisqr) + x;
            zi = (2 * zr * zi) + y;
            zr = new_zr;

            //If in a periodic orbit, assume it is trapped
            if (zr == 0.0 && zi == 0.0) {
                count = maxIterations;
            } else {
                zrsqr = zr * zr;
                zisqr = zi * zi;
                count++;
            }
        }

        final int tIndex = tY * getGlobalSize(0) + tX;
        iters[tIndex] = count;
        lastValuesR[tIndex] = zr;
        lastValuesI[tIndex] = zi;
        distancesR[tIndex] = dr;
        distancesI[tIndex] = di;

    }

    @Override
    public String toString() {
        return "Double";
    }
}
