package nimra.mandelexplorer;

import nimra.mandelexplorer.opencl.OpenCLDevice;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.ServiceLoader;

/**
 * Created: 29.01.20   by: Armin Haaf
 * Mandelbrot iterations per second -> Baseline is calculation of 1000x1000 image with max. iteration of 1000
 * for the typical mandelbrot set (-0,5,0) scale 2.5, escape 2, with 5x5 Tiles.
 *
 * @author Armin Haaf
 */
public class MipS {
    private int width = 1000;
    private int height = 1000;
    private int maxIter = 1000;
    private int tiles = 5;
    private long minTestDurationMillis = 5000;

    public long getMipS(ComputeDevice pComputeDevice, MandelImpl pMandelImpl) {
        return 1;
    }

    private long calcMipS(ComputeDevice pComputeDevice, MandelImpl pMandelImpl) {
        if (!pMandelImpl.supports(pComputeDevice)) {
            return -1;
        }
        final MandelParams tParams = new MandelParams();
        tParams.setMaxIterations(maxIter);
        tParams.setCalcMode(CalcMode.MANDELBROT);
        tParams.setX(new BigDecimal(-0.5));
        tParams.setY(new BigDecimal(-0));
        tParams.setScale(new BigDecimal(2.5));

        final MandelResult tMandelResult = new MandelResult(width, height);


        final TileGenerator tileGenerator = new TileGenerator();
        final List<Tile> tTilesList = tileGenerator.generateTiles(tMandelResult.width, tMandelResult.height, tiles);

        // warmup
        pMandelImpl.mandel(pComputeDevice, tParams, tMandelResult, tTilesList.get(0));

        try {
            Thread.sleep(1000);
        } catch (InterruptedException pE) {
            pE.printStackTrace();
        }

        final long tStart = System.currentTimeMillis();
        final long tEndTime = System.currentTimeMillis() + minTestDurationMillis;
        int tCount = 0;
        while (true) {
            tCount++;
            for (Tile tTile : tTilesList) {
                pMandelImpl.mandel(pComputeDevice, tParams, tMandelResult, tTile);
            }
            if (tEndTime <= System.currentTimeMillis()) {
                break;
            }
        }
        long tDuration = System.currentTimeMillis() - tStart;

        long tMipS = 0;
        for (int tIter : tMandelResult.iters) {
            tMipS += tIter;
        }
        return (tMipS * tCount * 1000) / tDuration;
    }

    public static void main(String[] args) {
        MipS tMipS = new MipS();
        ComputeDevice.DEVICES.toString();

        final List<MandelImpl> tImpls = new ArrayList<>();
        for (MandelImplFactory tMandelImplFactory : ServiceLoader.load(MandelImplFactory.class)) {
            tImpls.addAll(tMandelImplFactory.getMandelImpls());
        }
        for (MandelImpl tImpl : ServiceLoader.load(MandelImpl.class)) {
            tImpls.add(tImpl);
        }

        for (ComputeDevice tDevice : ComputeDevice.DEVICES) {
            if ( tDevice.getDeviceDescriptor() instanceof OpenCLDevice &&
                 ((OpenCLDevice)tDevice.getDeviceDescriptor()).getDeviceType() == OpenCLDevice.Type.CPU) {
                continue;
            }
            for (MandelImpl tImpl : tImpls) {
                try {
                    if (tImpl.supports(tDevice)) {
                        Thread.sleep(1000);
                        System.out.print(tDevice.getName() + ":" + tImpl + " = ");
                        System.out.println("" + tMipS.calcMipS(tDevice, tImpl) / 1024 / 1024);
                    }
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            }
        }
    }
}
