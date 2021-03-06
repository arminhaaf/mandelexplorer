package nimra.mandelexplorer.cuda;

import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static jcuda.driver.CUresult.CUDA_SUCCESS;
import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.runtime.JCuda.cudaGetDeviceCount;
import static jcuda.runtime.JCuda.cudaGetDeviceProperties;

/**
 * Created: 24.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class CudaDevice {
    private final CUdevice cuDevice;

    private final cudaDeviceProp deviceProperties;
    private final int deviceId;

    private CudaDevice(final int pDeviceId) {
        deviceId = pDeviceId;
        cuDevice = new CUdevice();
        cuDeviceGet(cuDevice, pDeviceId);

        CUcontext tContext = new CUcontext();
        cuCtxCreate(tContext, 0, cuDevice);

        deviceProperties = new cudaDeviceProp();
        cudaGetDeviceProperties(deviceProperties, pDeviceId);

        cuCtxDestroy(tContext);
    }

    public CUdevice getCuDevice() {
        return cuDevice;
    }

    public int getDeviceId() {
        return deviceId;
    }

    public String getName() {
        return deviceProperties.getName();
    }

    public String getVersion() {
        return deviceProperties.major + "." + deviceProperties.minor;
    }

    @Override
    public String toString() {
        return deviceId + " " + getName();
    }

    public int getWarpSize() {
        return deviceProperties.warpSize;
    }


    private static final List<CudaDevice> DEVICES;

    static {
        List<CudaDevice> tDevices = new ArrayList<>();
        try {
            if (cuInit(0) == CUDA_SUCCESS) {
                JCuda.setExceptionsEnabled(true);
                JCudaDriver.setExceptionsEnabled(true);
                int tDeviceCount[] = {0};
                cudaGetDeviceCount(tDeviceCount);
                for (int tDeviceId = 0; tDeviceId < tDeviceCount[0]; tDeviceId++) {
                    CudaDevice tCudaDevice = new CudaDevice(tDeviceId);
                    System.out.println("found cuda device " + tCudaDevice);
                    tDevices.add(tCudaDevice);
                }
            }
        } catch (Throwable ex) {
            ex.printStackTrace();
        }
        DEVICES = Collections.unmodifiableList(tDevices);
    }

    public static List<CudaDevice> getDevices() {
        return DEVICES;
    }


    public static void main(String[] args) {
        System.out.println("test");
    }

}
