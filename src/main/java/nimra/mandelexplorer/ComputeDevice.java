package nimra.mandelexplorer;

/**
 * Created: 15.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class ComputeDevice {
    public static ComputeDevice CPU = new ComputeDevice("CPU");

    private String name;

    private Object deviceDescriptor;

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

    @Override
    public String toString() {
        return name;
    }
}
