package nimra.mandelexplorer;

import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;
import org.json.JSONObject;

import java.lang.reflect.Field;
import java.math.BigDecimal;

import static java.math.MathContext.DECIMAL128;

/**
 * Created: 27.12.19   by: Armin Haaf
 * <p>
 * mandelbrot calc parameters
 *
 * @author Armin Haaf
 */
public class MandelParams {
    private BigDecimal x = new BigDecimal(-1);
    private BigDecimal y = new BigDecimal(0);
    private BigDecimal scale = new BigDecimal(3);

    private int maxIterations = 100;
    private double escapeRadius = 2;

    public double getX_Double() {
        return x.doubleValue();
    }

    public double getY_Double() {
        return y.doubleValue();
    }

    public double getScale_double() {
        return scale.doubleValue();
    }

    public BigDecimal getX() {
        return x;
    }

    public void setX(BigDecimal x) {
        this.x = x;
    }

    public BigDecimal getY() {
        return y;
    }

    public void setY(BigDecimal y) {
        this.y = y;
    }

    public BigDecimal getScale() {
        return scale;
    }

    public void setScale(BigDecimal scale) {
        this.scale = scale;
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
        final JSONObject tJSONObject = new JSONObject();
        tJSONObject.put("x", x.toString());
        tJSONObject.put("y", y.toString());
        tJSONObject.put("scale", scale.toString());
        tJSONObject.put("maxIterations", maxIterations);
        tJSONObject.put("escapeRadius", escapeRadius);
        return tJSONObject;
    }

    public MandelParams fromJson(JSONObject pJSONObject) {
        try {
            for (Field tField : getClass().getDeclaredFields()) {
                tField.setAccessible(true);
                if (pJSONObject.has(tField.getName())) {
                    final Object tValue = pJSONObject.get(tField.getName());
                    if (BigDecimal.class == tField.getType() && tValue!=JSONObject.NULL) {
                        if ( tValue instanceof String ) {
                            tField.set(this, new BigDecimal((String) tValue, DECIMAL128));
                        } else if ( tValue instanceof Double ) {
                            tField.set(this, new BigDecimal(Double.toString((Double) tValue), DECIMAL128));
                        }
                    } else {
                        tField.set(this, tValue == JSONObject.NULL ? null : tValue);
                    }
                }
            }
        } catch (IllegalAccessException pE) {
            pE.printStackTrace();
        }

        return this;
    }

}
