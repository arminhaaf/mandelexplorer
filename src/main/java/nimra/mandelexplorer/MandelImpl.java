package nimra.mandelexplorer;

import nimra.mandelexplorer.palette.DefaultPaletteMapper;

import java.math.BigDecimal;

/**
 * Created: 12.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public interface MandelImpl {
    void mandel(final ComputeDevice pComputeDevice, MandelParams pParams, MandelResult pMandelResult, Tile pTile);

    /**
     * the device to calc
     *
     * @return true, if calc device is supported
     */
    default boolean supports(ComputeDevice pDevice) {
        return pDevice == ComputeDevice.CPU;
    }

    default boolean isPreciseFor(BigDecimal pPixelSize) {
        return true;
    }

    default boolean supportsMode(CalcMode pMode) {
        return true;
    }

    default MandelParams getHomeParams() {
        return new MandelParams();
    }

    default PaletteMapper getDefaultPaletteMapper() {
        return new DefaultPaletteMapper();
    }

    /**
     * For multi device computation. If not threadsafe the impl is copied for every tile.
     *
     * @return
     */
    default boolean isThreadSafe() {
        return true;
    }

    /**
     * should be overriden for not thread sage implementations
     *
     * @return
     */
    default MandelImpl copy() {
        return this;
    }

    /**
     * called after calculation
     */
    default void done() {
    }

}
