package nimra.mandelexplorer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created: 15.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class ComputeDevice {
    public static ComputeDevice CPU = new ComputeDevice("CPU");

    public static final List<ComputeDevice> DEVICES;

    static {
        final List<ComputeDevice> tDevices = new ArrayList<>();
        tDevices.add(ComputeDevice.CPU);
        for (nimra.mandelexplorer.opencl.OpenCLDevice tDevice : nimra.mandelexplorer.opencl.OpenCLDevice.getDevices()) {
            tDevices.add(new ComputeDevice("OpenCL " + tDevice.getName(), tDevice));
        }

        for (nimra.mandelexplorer.cuda.CudaDevice tDevice : nimra.mandelexplorer.cuda.CudaDevice.getDevices()) {
            tDevices.add(new ComputeDevice("Cuda " + tDevice.getName(), tDevice));
        }

        DEVICES = Collections.unmodifiableList(tDevices);
    }

    private String name;

    private Object deviceDescriptor;

    private boolean enabled = true;

    public ComputeDevice(final String pName) {
        name = pName;
    }

    public ComputeDevice(final String pName, final Object pDeviceDescriptor) {
        name = pName;
        deviceDescriptor = pDeviceDescriptor;
    }

    public String getName() {
        return name;
    }

    public Object getDeviceDescriptor() {
        return deviceDescriptor;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(final boolean pEnabled) {
        enabled = pEnabled;
    }

    @Override
    public String toString() {
        return name;
    }
}
