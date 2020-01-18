package nimra.mandelexplorer.old;

import com.aparapi.Kernel;
import com.aparapi.device.Device;
import com.aparapi.device.OpenCLDevice;
import com.aparapi.opencl.OpenCL;
import org.apache.commons.lang3.tuple.Pair;

import java.lang.ref.SoftReference;
import java.util.HashMap;
import java.util.Map;

public class CLImplCache {

    private static final Map<Pair<Kernel, Device>, SoftReference<? extends OpenCL<?>>> implCache = new HashMap<>();

    public static <E extends OpenCL<E>> E getImpl(Kernel pKernel, Class<E> pStub) {
        Device tTargetDevice = pKernel.getTargetDevice();
        if ( !(tTargetDevice instanceof OpenCLDevice)) {
            throw new RuntimeException("need OpenCL device to run " + pKernel.toString());
        }
        SoftReference<E> tRef = (SoftReference<E>) implCache.get(Pair.of(pKernel, tTargetDevice));
        E tImpl = tRef != null ? tRef.get() : null;
        if (tImpl == null) {
            tImpl = ((OpenCLDevice) tTargetDevice).bind(pStub);
            implCache.put(Pair.of(pKernel, tTargetDevice), new SoftReference<>(tImpl));
            return tImpl;
        } else {
            return tImpl;
        }

    }
}
