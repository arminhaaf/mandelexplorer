package nimra.mandelexplorer.opencl;

import nimra.mandelexplorer.ComputeDevice;
import nimra.mandelexplorer.MandelParams;
import nimra.mandelexplorer.MandelResult;
import nimra.mandelexplorer.Tile;
import nimra.mandelexplorer.TileGenerator;

import java.util.List;
import java.util.Scanner;

/**
 * Created: 18.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class TestOpenCLMandelImpl {

    public static void main(String[] args) {
        OpenCLMandelImpl tCLMandel = new OpenCLMandelImpl();
        tCLMandel.setCode(new Scanner(OpenCLDevice.class.getResourceAsStream("/DoubleMandel.cl"), "UTF-8").useDelimiter("\\A").next());
        for (OpenCLDevice tOpenCLDevice : OpenCLDevice.getDevices()) {
            System.out.println(tOpenCLDevice);

            try {
                final ComputeDevice tComputeDevice = new ComputeDevice(tOpenCLDevice.getName(), tOpenCLDevice);
                tCLMandel.supports(tComputeDevice);

                MandelResult tMandelResult = new MandelResult(48, 24);
                final MandelParams tParams = new MandelParams();
                tParams.setMaxIterations(10);

                TileGenerator tTileGenerator = new TileGenerator();
                List<Tile> tTiles = tTileGenerator.generateTiles(tMandelResult.width, tMandelResult.height, 1);
                for (int i = 0; i < 1; i++) {
                    long tStartMillis = System.currentTimeMillis();
                    try {
                        for (Tile tTile : tTiles) {
                            tCLMandel.mandel(tComputeDevice, tParams, tMandelResult, tTile);
                        }
                    } catch (Exception ex) {
                        ex.printStackTrace();
                    }
                    System.out.println("duration loop " + i + " " + (System.currentTimeMillis() - tStartMillis));
                    for (int y = 0; y < tMandelResult.height; y++) {
                        for (int x = 0; x < tMandelResult.width; x++) {
                            if (tMandelResult.iters[x + y * tMandelResult.width] < tParams.getMaxIterations()) {
                                System.out.print(".");
                            } else {
                                System.out.print("x");
                            }
                        }
                        System.out.println();
                    }
                }


            } catch (Exception ex) {
                ex.printStackTrace();
            }

            return;

        }
    }


}
