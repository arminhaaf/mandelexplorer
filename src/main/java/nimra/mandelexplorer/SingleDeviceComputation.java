package nimra.mandelexplorer;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Created: 16.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class SingleDeviceComputation implements Computation {

    private final TileListener tileListener;

    private final TileGenerator tileGenerator = new TileGenerator();

    public SingleDeviceComputation(final TileListener pTileListener) {
        tileListener = pTileListener;
    }

    @Override
    public void compute(
            final AtomicBoolean pCancel,
            final MandelImpl pImpl,
            final MandelParams pMandelParams,
            final MandelResult pMandelResult,
            int tTileCount) {

        final List<Tile> tTilesList = tileGenerator.generateTiles(pMandelResult.width, pMandelResult.height, tTileCount);

        try {
            for (ComputeDevice tComputeDevice : ComputeDevice.DEVICES) {
                if (tComputeDevice.isEnabled() && pImpl.supports(tComputeDevice)) {
                    for (final Tile tTile : tTilesList) {
                        pImpl.mandel(tComputeDevice, pMandelParams, pMandelResult, tTile);
                        tileListener.tileReady(tTile);
                        if (pCancel.get()) {
                            return;
                        }
                    }
                    return;
                }
            }
        } finally {
            pImpl.done();
        }

    }

}
