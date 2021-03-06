package nimra.mandelexplorer.opencl;

import org.jocl.CL;
import org.jocl.NativePointerObject;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_device_id;
import org.jocl.cl_platform_id;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.jocl.CL.CL_DEVICE_TYPE_ALL;
import static org.jocl.CL.clGetDeviceIDs;
import static org.jocl.CL.clGetDeviceInfo;
import static org.jocl.CL.clGetPlatformIDs;

/**
 * Created: 16.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class OpenCLDevice {
    private final cl_device_id clDeviceId;

    private final String name;
    private final String vendor;
    private final String version;

    private final Type deviceType;

    // the native pointer
    private final long deviceId;

    private OpenCLDevice(final cl_device_id pClDeviceId) {
        clDeviceId = pClDeviceId;

        name = getString(clDeviceId, CL.CL_DEVICE_NAME);
        vendor = getString(clDeviceId, CL.CL_DEVICE_VENDOR);
        version = getString(clDeviceId, CL.CL_DEVICE_VERSION);

        long tDeviceType = getLong(clDeviceId, CL.CL_DEVICE_TYPE);
        if ((tDeviceType & CL.CL_DEVICE_TYPE_CPU) != 0) {
            deviceType = Type.CPU;
        } else if ((tDeviceType & CL.CL_DEVICE_TYPE_GPU) != 0) {
            deviceType = Type.GPU;
        } else if ((tDeviceType & CL.CL_DEVICE_TYPE_ACCELERATOR) != 0) {
            deviceType = Type.ACCELERATOR;
        } else {
            deviceType = Type.UNKNOWN;
        }

        // we need the device id to match with aparapi device -> no public getter...
        long tDeviceId = 0;
        try {
            final Method tMethod = NativePointerObject.class.getDeclaredMethod("getNativePointer");
            tMethod.setAccessible(true);
            tDeviceId = (long)tMethod.invoke(clDeviceId);
        } catch (Exception pE) {
            pE.printStackTrace();
        }
        deviceId = tDeviceId;
    }

    private String getString(cl_device_id device, int paramName) {
        // Obtain the length of the string that will be queried
        long[] size = new long[1];
        clGetDeviceInfo(device, paramName, 0, null, size);

        // Create a buffer of the appropriate size and fill it with the info
        byte[] buffer = new byte[(int)size[0]];
        clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);

        // Create a string from the buffer (excluding the trailing \0 byte)
        return new String(buffer, 0, buffer.length - 1);
    }

    public cl_device_id getCLDeviceId() {
        return clDeviceId;
    }

    public long getDeviceId() {
        return deviceId;
    }

    public String getName() {
        return name;
    }

    public String getVendor() {
        return vendor;
    }

    public String getVersion() {
        return version;
    }

    public Type getDeviceType() {
        return deviceType;
    }

    private static long getLong(cl_device_id device, int paramName) {
        return getLongs(device, paramName, 1)[0];
    }

    private static long[] getLongs(cl_device_id device, int paramName, int numValues) {
        long[] values = new long[numValues];
        clGetDeviceInfo(device, paramName, Sizeof.cl_long * numValues, Pointer.to(values), null);
        return values;
    }

    @Override
    public String toString() {
        return clDeviceId.toString() + " " + name;
    }


    private static final List<OpenCLDevice> DEVICES;

    static {
        final List<OpenCLDevice> tDevices = new ArrayList<>();
        try {
            // Obtain the number of platforms
            final int[] numPlatforms = new int[1];
            clGetPlatformIDs(0, null, numPlatforms);
            // Obtain the platform IDs
            final cl_platform_id[] platforms = new cl_platform_id[numPlatforms[0]];
            clGetPlatformIDs(platforms.length, platforms, null);

            for (final cl_platform_id pPlatform : platforms) {
                // Obtain the number of devices for the current platform
                final int[] numDevices = new int[1];
                clGetDeviceIDs(pPlatform, CL_DEVICE_TYPE_ALL, 0, null, numDevices);
                final cl_device_id[] devicesArray = new cl_device_id[numDevices[0]];
                clGetDeviceIDs(pPlatform, CL_DEVICE_TYPE_ALL, numDevices[0], devicesArray, null);

                for (cl_device_id tCl_device_id : devicesArray) {
                    final OpenCLDevice tOpenCLDevice = new OpenCLDevice(tCl_device_id);
                    System.out.println("found " + tOpenCLDevice);
                    tDevices.add(tOpenCLDevice);
                }
            }

        } catch ( Throwable ex) {
            ex.printStackTrace();
        }
        DEVICES = Collections.unmodifiableList(tDevices);
    }

    public static List<OpenCLDevice> getDevices() {
        return DEVICES;
    }


    public enum Type {
        UNKNOWN,
        GPU,
        CPU,
        ACCELERATOR
    }
}
