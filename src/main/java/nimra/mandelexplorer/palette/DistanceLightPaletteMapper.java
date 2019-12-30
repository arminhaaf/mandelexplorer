package nimra.mandelexplorer.palette;

import nimra.mandelexplorer.PaletteMapper;
import org.json.JSONObject;

import java.awt.Color;

/**
 * Created: 27.12.19   by: Armin Haaf
 *
 * from https://www.math.univ-toulouse.fr/~cheritat/wiki-draw/index.php/Mandelbrot_set
 *
 * @author Armin Haaf
 */
public class DistanceLightPaletteMapper extends PaletteMapper {

    double height = 1.5; // height factor of the incoming light
    double angle = 45.0; // incoming direction of light in turns
    boolean inverse = false;
    Complex v;


    public DistanceLightPaletteMapper() {
        insideColor = new Color(0x2c0091);
    }

    @Override
    public void startMap() {
        super.startMap();
        v = new Complex(0, 2.0 * angle / 360.0 * Math.PI).exp(); // = exp(1j*angle*2*pi/360)  // unit 2D vector in this direction
    }

    @Override
    public int map(final int pIter, final double pLastR, final double pLastI, final double pDistanceR, final double pDistanceI) {


        if (pIter != mandelParams.getMaxIterations()) {
            final Complex z = new Complex(pLastR, pLastI);
            final Complex der = new Complex(pDistanceR, pDistanceI);

            Complex u = z.divides(der);
            u = u.scale(1 / u.abs());

            double tReflection = u.re() * v.re() + u.im() * v.im() + height;
            tReflection = tReflection / (1 + height);

            if (tReflection < 0) {
                tReflection = 0;
            }
//
            return Color.HSBtoRGB(0, 0, inverse ? (float)tReflection : 1-(float)tReflection);
        } else {
            return insideColor.getRGB();
        }
    }

    @Override
    public String getName() {
        return "DistanceLight";
    }

    @Override
    protected void toJson(final JSONObject pJSONObject) {
        super.toJson(pJSONObject);
        pJSONObject.put("height", height);
        pJSONObject.put("angle", angle);
        pJSONObject.put("inverse", inverse);
    }

    @Override
    public void fromJson(final JSONObject pJSONObject) {
        super.fromJson(pJSONObject);
        height = pJSONObject.optDouble("height", 1.5);
        angle = pJSONObject.optDouble("angle", 45);
        inverse = pJSONObject.optBoolean("inverse", false);
    }


}
