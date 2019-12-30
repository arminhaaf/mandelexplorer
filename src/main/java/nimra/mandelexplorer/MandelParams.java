package nimra.mandelexplorer;

import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;
import org.json.JSONObject;

import java.lang.reflect.Field;

/**
 * Created: 27.12.19   by: Armin Haaf
 *
 * mandelbrot calc parameters
 *
 * @author Armin Haaf
 */
public class MandelParams {
    private double x = -1f;
    private double y = 0f;
    private double scale = 3;
    private int maxIterations = 100;
    private double escapeRadius = 2;

    public double getX() {
        return x;
    }

    public void setX(final double pX) {
        x = pX;
    }

    public double getY() {
        return y;
    }

    public void setY(final double pY) {
        y = pY;
    }

    public double getScale() {
        return scale;
    }

    public void setScale(final double pScale) {
        scale = pScale;
    }

    public int getMaxIterations() {
        return maxIterations;
    }

    public void setMaxIterations(final int pMaxIterations) {
        maxIterations = pMaxIterations;
    }

    public double getEscapeRadius() {
        return escapeRadius;
    }

    public void setEscapeRadius(final double pEscapeRadius) {
        escapeRadius = pEscapeRadius;
    }

    @Override
    public String toString() {
        return ToStringBuilder.reflectionToString(this, ToStringStyle.JSON_STYLE);
    }

    public JSONObject toJson() {
        return new JSONObject(toString());
    }

    public MandelParams fromJson(JSONObject pJSONObject) {
        try {
            for (Field tField : getClass().getDeclaredFields()) {
                tField.setAccessible(true);
                if ( pJSONObject.has(tField.getName())) {
                    final Object tValue = pJSONObject.get(tField.getName());
                    tField.set(this, tValue == JSONObject.NULL ? null : tValue);
                }
            }
        } catch (IllegalAccessException pE) {
            pE.printStackTrace();
        }

        return this;
    }
}
