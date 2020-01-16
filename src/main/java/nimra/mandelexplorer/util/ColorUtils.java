package nimra.mandelexplorer.util;

import org.beryx.awt.color.ColorFactory;

import java.awt.Color;
import java.lang.reflect.Field;

/**
 * Created: 16.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class ColorUtils {
    public static String toColorString(Color pColor) {
        for (Field tDeclaredField : ColorFactory.class.getDeclaredFields()) {
            if (Color.class.equals(tDeclaredField.getType())) {
                try {
                    Color tColor = (Color)tDeclaredField.get(null);
                    if (pColor.equals(tColor)) {
                        return tDeclaredField.getName().toLowerCase();
                    }
                } catch (IllegalAccessException pE) {
                    pE.printStackTrace();
                }
            }
        }
        return "#" + String.format("%06X", pColor.getRGB() & 0xffffff);
    }

}
