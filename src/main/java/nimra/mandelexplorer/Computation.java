package nimra.mandelexplorer;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Created: 16.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public interface Computation {
    void compute(
            AtomicBoolean pCancel,
            MandelImpl pImpl,
            MandelParams pMandelParams,
            MandelResult pMandelResult,
            int tTileCount);

    interface TileListener {
        void tileReady(Tile pTile);
    }
}
