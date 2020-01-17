package nimra.mandelexplorer;

import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;

/**
 * Created: 16.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public final class Tile {
    public final int startX, startY, endX, endY;

    public Tile(final int pStartX, final int pStartY, final int pEndX, final int pEndY) {
        startX = pStartX;
        startY = pStartY;
        endX = pEndX;
        endY = pEndY;
    }

    public int getWidth() {
        return endX - startX;
    }

    public int getHeight() {
        return endY - startY;
    }

    @Override
    public String toString() {
        return ToStringBuilder.reflectionToString(this, ToStringStyle.JSON_STYLE);
    }
}
