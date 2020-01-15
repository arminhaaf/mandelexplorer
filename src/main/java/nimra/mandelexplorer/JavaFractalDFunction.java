package nimra.mandelexplorer;

/**
 * Created: 12.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class JavaFractalDFunction implements FractalDFunction {

    private static final JavaCodeManager CODE_MANAGER = new JavaCodeManager();

    private final String script;

    private final String name;

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

    private FractalDFunction compile() {
        return null;
    }

    @Override
    public void calc(final ComplexD tResult, final ComplexD z, final ComplexD c) {
        fractalDFunction.calc(tResult, z, c);
    }

    public String getSource() {
        return "public class " + name + " implements nimra.mandelexplorer.FractalDFunction {" +
               "    public void calc(final nimra.mandelexplorer.ComplexD tResult, final nimra.mandelexplorer.ComplexD z, final nimra.mandelexplorer.ComplexD c) {\n" +
               script +
               "    }\n" +
               "    public String toString() {\n" +
               "        return \"" + name + "\";\n" +
               "    }" +
               "}\n";
    }
}
