package nimra.mandelexplorer;

import com.aparapi.device.Device;

import java.math.BigDecimal;

/**
 * Created: 12.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public interface MandelImpl {
    void mandel(MandelParams pParams, int width, int height,
            int startX, int endX, int startY, int endY,
            boolean pCalcDistance,
            MandelResult pMandelResult);

    default boolean isCompatible(Device pDevice) {
        return true;
    }

    default boolean isPreciseFor(BigDecimal pPixelSize) {
        return true;
    }
}
