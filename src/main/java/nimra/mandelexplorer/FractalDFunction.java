package nimra.mandelexplorer;

import nimra.mandelexplorer.math.ComplexD;

/**
 * Created: 15.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public interface FractalDFunction {
    void calc(ComplexD tResult, ComplexD z, ComplexD c);
}
