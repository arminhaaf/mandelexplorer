package nimra.mandelexplorer;

import nimra.mandelexplorer.opencl.OpenCLDevice;

/**
 * Created: 18.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class FastDDMandelImpl implements MandelImpl {
    // avx2 dd und opencl dd für nicht CPU

    private final MandelImpl cpuImpl = new DDMandelNative(DDMandelNative.Algo.AVXDoubleDouble);
    private final MandelImpl gpuImpl = new DDOpenCLMandelImpl();

    @Override
    public boolean supports(final ComputeDevice pDevice) {
        if ( pDevice==ComputeDevice.CPU) {
            return true;
        }
        if ( pDevice.getDeviceDescriptor() instanceof OpenCLDevice ) {
            return ((OpenCLDevice)pDevice.getDeviceDescriptor()).getDeviceType() != OpenCLDevice.Type.CPU;
        }
        return true;
    }

    @Override
    public void mandel(final ComputeDevice pComputeDevice, final MandelParams pParams, final MandelResult pMandelResult, final Tile pTile) {
        if ( pComputeDevice==ComputeDevice.CPU) {
            cpuImpl.mandel(pComputeDevice, pParams, pMandelResult, pTile);
        } else {
            gpuImpl.mandel(pComputeDevice, pParams, pMandelResult, pTile);
        }
    }

    @Override
    public boolean isAvailable() {
        return gpuImpl.isAvailable() && cpuImpl.isAvailable();
    }

    @Override
    public String toString() {
        return "Fast DD";
    }
}
