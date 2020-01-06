package nimra.mandelexplorer;

import java.util.function.Supplier;

public class MandelAlgo {

    private final String name;
    private final Supplier<MandelKernel> mandelKernelSupplier;

    private double precision;

    private MandelKernel mandelKernel;

    public MandelAlgo(String pName, Supplier<MandelKernel> pMandelKernelSupplier) {
        this.name = pName;
        this.mandelKernelSupplier = pMandelKernelSupplier;
    }

    public MandelAlgo(String name, double precision, Supplier<MandelKernel> mandelKernelSupplier) {
        this.name = name;
        this.mandelKernelSupplier = mandelKernelSupplier;
        this.precision = precision;
    }

    public synchronized  MandelKernel getMandelKernel() {
        if ( mandelKernel==null) {
            mandelKernel = mandelKernelSupplier.get();
        }
        return  mandelKernel;
    }

    @Override
    public String toString() {
        return name;
    }

    public synchronized void dispose() {
        if ( mandelKernel!=null ) {
            mandelKernel.dispose();
            mandelKernel = null;
        }
    }

    public double getPrecision() {
        return precision;
    }
}
