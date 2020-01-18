package nimra.mandelexplorer;

import nimra.mandelexplorer.opencl.OpenCLDevice;

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

                MandelResult tMandelResult = new MandelResult(1000, 1000);
                final MandelParams tParams = new MandelParams();
                tParams.setMaxIterations(1000);

                TileGenerator tTileGenerator = new TileGenerator();
                List<Tile> tTiles = tTileGenerator.generateTiles(tMandelResult.width, tMandelResult.height, 5);
                for (int i = 0; i < 5; i++) {
                    long tStartMillis = System.currentTimeMillis();
                    for (Tile tTile : tTiles) {
                        try {
                            tCLMandel.mandel(tComputeDevice, tParams, tMandelResult, tTile);
                        } catch (Exception ex) {
                            ex.printStackTrace();
                        }
                    }
                    System.out.println("duration loop " + i + " " + (System.currentTimeMillis() - tStartMillis));
                }

            } catch (Exception ex) {
                ex.printStackTrace();
            }


        }
    }


}
