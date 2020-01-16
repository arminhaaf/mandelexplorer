package nimra.mandelexplorer;

/**
 * Created: 12.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class MandelResult {
    public final int width;

    public final int height;

    public final int iters[];

    public final double lastValuesR[];

    public final double lastValuesI[];

    public final double distancesR[];

    public final double distancesI[];

    public MandelResult(int pWidth, int pHeight) {
        width = pWidth;
        height = pHeight;
        iters = new int[pWidth*pHeight];
        lastValuesR = new double[pWidth*pHeight];
        lastValuesI = new double[pWidth*pHeight];
        distancesR = new double[pWidth*pHeight];
        distancesI = new double[pWidth*pHeight];
    }
}
