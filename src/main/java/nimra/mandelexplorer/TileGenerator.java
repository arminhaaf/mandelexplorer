package nimra.mandelexplorer;

import java.util.ArrayList;
import java.util.List;

/**
 * Created: 15.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class TileGenerator {

    public List<Tile> generateTiles(int pWidth, int pHeight, int pTileCount) {
        final List<Tile> tTiles = new ArrayList<>();

        final int tMinTileWidth = pWidth / pTileCount;
        final int tMinTileHeight = pHeight / pTileCount;
        int tTileHeightIncrease = pHeight % pTileCount;
        int tEndY = 0;
        for (int y = 0; y < pTileCount; y++) {
            int tStartY = tEndY;
            tEndY = tStartY + tMinTileHeight + (tTileHeightIncrease-- > 0 ? 1 : 0);

            int tEndX = 0;
            int tTileWidthIncrease = pWidth % pTileCount;
            for (int x = 0; x < pTileCount; x++) {
                int tStartX = tEndX;
                tEndX = tStartX + tMinTileWidth + (tTileWidthIncrease-- > 0 ? 1 : 0);

                tTiles.add(new Tile(tStartX, tStartY, tEndX, tEndY));
            }
        }

        return tTiles;
    }

}
