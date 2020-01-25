package nimra.mandelexplorer;

import nimra.mandelexplorer.math.ComplexD;
import org.json.JSONObject;

/**
 * Created: 12.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class JavaFractalDFunction implements FractalDFunction {

    private static final JavaCodeManager CODE_MANAGER = new JavaCodeManager();

    private String script;

    private String name;

    private String description;

    private final FractalDFunction fractalDFunction;

    public JavaFractalDFunction(final String pScript, final String pName) {
        script = pScript;
        name = pName;

        FractalDFunction tImpl = null;
        try {
            CODE_MANAGER.compile(pName, getSource());
            tImpl = CODE_MANAGER.newInstance(pName, FractalDFunction.class);
        } catch (Exception pE) {
            pE.printStackTrace();
        }
        fractalDFunction = tImpl;
    }

    public String getScript() {
        return script;
    }

    public String getName() {
        return name;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(final String pDescription) {
        description = pDescription;
    }

    private FractalDFunction compile() {
        return null;
    }

    @Override
    public void calc(final ComplexD tResult, final ComplexD z, final ComplexD c) {
        fractalDFunction.calc(tResult, z, c);
    }

    public String getSource() {
        return "import nimra.mandelexplorer.ComplexD;\n" +
               "public class " + name + " implements nimra.mandelexplorer.FractalDFunction {" +
               "    public void calc(final ComplexD tResult, final ComplexD z, final ComplexD c) {\n" +
               script +
               "    }\n" +
               "    public String toString() {\n" +
               "        return \"" + name + "\";\n" +
               "    }" +
               "}\n";
    }

    public JSONObject toJson() {
        final JSONObject tJSONObject = new JSONObject();
        tJSONObject.put("name", name);
        tJSONObject.put("description", description);
        tJSONObject.put("script", script);
        return tJSONObject;
    }

    public JavaFractalDFunction fromJson(JSONObject pJSONObject) {
        name = pJSONObject.getString("name");
        description = pJSONObject.optString("description", null);
        script = pJSONObject.getString("script");
        return this;
    }

}
