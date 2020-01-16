package nimra.mandelexplorer;

import nimra.mandelexplorer.util.SimplePool;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;

/**
 * Created: 16.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class MultiDeviceComputation {

    private final TileListener tileListener;

    private final TileGenerator tileGenerator = new TileGenerator();

    private final Map<MandelImpl, SimplePool<MandelImpl>> implPools = new HashMap<>();

    public MultiDeviceComputation(final TileListener pTileListener) {
        tileListener = pTileListener;
    }

    private synchronized SimplePool<MandelImpl> getImplPool(MandelImpl pImpl) {
        SimplePool<MandelImpl> tPool = implPools.get(pImpl);
        if (tPool == null) {
            tPool = new MandelImplPool(pImpl::copy);
            implPools.put(pImpl, tPool);
        }
        return tPool;
    }


    private MandelImpl getCalcInstance(MandelImpl pMandelImpl) {
        if (pMandelImpl.isThreadSafe()) {
            return pMandelImpl;
        } else {
            return getImplPool(pMandelImpl).borrow();
        }
    }


    public void compute(
            final AtomicBoolean pCancel,
            final MandelImpl pImpl,
            final List<ComputeDevice> enabledDevices,
            final MandelParams pMandelParams,
            final MandelResult pMandelResult,
            int tTileCount) {

        final List<Tile> tTilesList = tileGenerator.generateTiles(pMandelResult.width, pMandelResult.height, tTileCount);

        final List<Thread> tThreads = new ArrayList<>();
        for (ComputeDevice tEnabledDevice : enabledDevices) {
            final MandelImpl tMandelImpl = getCalcInstance(pImpl);
            if (tMandelImpl.setComputeDevice(tEnabledDevice)) {
                Thread tCalcThread = new Thread(() -> {
                    while (true) {
                        Tile tTile = null;
                        synchronized (tTilesList) {
                            if (tTilesList.size() > 0) {
                                tTile = tTilesList.remove(0);
                            }
                        }
                        if (pCancel.get() || tTile == null) {
                            // a new calculation is requested
                            tMandelImpl.done();
                            return;
                        }
                        tMandelImpl.mandel(pMandelParams, pMandelResult, tTile);
                        tileListener.tileReady(tTile);
                    }
                });
                tCalcThread.setName("Calc Thread " + tEnabledDevice.getName());
                tThreads.add(tCalcThread);
                tCalcThread.start();
            }
        }

        for (Thread tThread : tThreads) {
            try {
                tThread.join();
            } catch (InterruptedException pE) {
                pE.printStackTrace();
            }
        }

    }


    public interface TileListener {
        void tileReady(Tile pTile);
    }


    private class MandelImplPool extends SimplePool<MandelImpl> {

        public MandelImplPool(final Supplier<MandelImpl> pFactory) {
            super(pFactory);
        }

        @Override
        protected MandelImpl createInstance() {
            return new MandelImplPoolProxy(super.createInstance());
        }

        private class MandelImplPoolProxy implements MandelImpl {
            private final MandelImpl impl;

            public MandelImplPoolProxy(final MandelImpl pInstance) {
                impl = pInstance;
            }

            @Override
            public void mandel(final MandelParams pParams, final MandelResult pMandelResult, final Tile pTile) {
                impl.mandel(pParams, pMandelResult, pTile);
            }

            @Override
            public boolean setComputeDevice(final ComputeDevice pDevice) {
                return impl.setComputeDevice(pDevice);
            }

            @Override
            public boolean isPreciseFor(final BigDecimal pPixelSize) {
                return impl.isPreciseFor(pPixelSize);
            }

            @Override
            public boolean supportsMode(final CalcMode pMode) {
                return impl.supportsMode(pMode);
            }

            @Override
            public MandelParams getHomeParams() {
                return impl.getHomeParams();
            }

            @Override
            public PaletteMapper getDefaultPaletteMapper() {
                return impl.getDefaultPaletteMapper();
            }

            @Override
            public boolean isThreadSafe() {
                return impl.isThreadSafe();
            }

            @Override
            public MandelImpl copy() {
                return impl.copy();
            }

            @Override
            public void done() {
                MandelImplPool.this.done(this);
            }
        }

    }


}
