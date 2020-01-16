package nimra.mandelexplorer;

/**
 * Created: 16.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public
enum CalcMode {
    MANDELBROT(1), MANDELBROT_DISTANCE(2), JULIA(3);

    private int modeNumber;

    private CalcMode(final int pModeNumber) {
        modeNumber = pModeNumber;
    }

    public int getModeNumber() {
        return modeNumber;
    }
}
