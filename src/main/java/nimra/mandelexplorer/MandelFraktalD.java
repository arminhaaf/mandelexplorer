package nimra.mandelexplorer;

/**
 * Created: 15.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class MandelFraktalD implements FractalDFunction {
    @Override
    public void calc(final ComplexD tResult, final ComplexD z, final ComplexD c) {
        // z^2 + c;
        tResult.set(z).sqr().add(c);
    }
}
