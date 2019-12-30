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

import com.aparapi.Kernel;

import java.util.ArrayList;
import java.util.List;

/**
 * Created: 26.12.19   by: Armin Haaf
 *
 * A simple abstract class for different algoritms -> it cold be shared a lot more code, however aparapi is restricted
 *
 * @author Armin Haaf
 */
public abstract class MandelKernel extends Kernel {

    // dispose the kernels at the end
    private static final List<Kernel> KERNELS = new ArrayList<>();

    protected final int width;
    protected final int height;

    /**
     * buffer used to store the iterations (width * height).
     */
    final protected int iters[];

    /**
     * buffer used to store the last calculated real value of a point -> used for some palette calculations
     */
    final protected double lastValuesR[];

    /**
     * buffer used to store the last calculated imaginary value of a point -> used for some palette calculations
     */
    final protected double lastValuesI[];

    final protected double distancesR[];
    final protected double distancesI[];


    public MandelKernel(int pWidth, int pHeight) {
        width = pWidth;
        height = pHeight;
        iters = new int[pWidth * pHeight];
        lastValuesR = new double[pWidth * pHeight];
        lastValuesI = new double[pWidth * pHeight];
        distancesR = new double[pWidth * pHeight];
        distancesI = new double[pWidth * pHeight];

        KERNELS.add(this);
    }


    // prepare the calculation
    public abstract void init(final MandelParams pMandelParams);

    public int[] getIters() {
        return iters;
    }

    public double[] getLastValuesR() {
        return lastValuesR;
    }

    public double[] getLastValuesI() {
        return lastValuesI;
    }

    public double[] getDistancesR() {
        return distancesR;
    }

    public double[] getDistancesI() {
        return distancesI;
    }


    public static void disposeAll() {
        for (Kernel tKernel : KERNELS) {
            tKernel.dispose();
        }
        KERNELS.clear();
    }
}
