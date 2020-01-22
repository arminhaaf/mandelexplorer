package nimra.mandelexplorer.util;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.HashSet;
import java.util.Set;

/**
 * Created: 20.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class NativeLoader {
    private static final String OS_NAME = System.getProperty("os.name").toLowerCase();
    private static final String OS_ARCH = System.getProperty("os.arch").toLowerCase();

    private static Set<String> LOADEDLIBS = new HashSet<>();

    public static synchronized void loadNativeLib(String pLib) {
        if ( LOADEDLIBS.contains(pLib)) {
            return;
        }

        LOADEDLIBS.add(pLib);

        final String tSystemLibName = System.mapLibraryName(pLib);

        String tSystemDir = "unknown";
        final boolean t64Bit = OS_ARCH.contains("64");

        if (OS_NAME.startsWith("linux")) {
            tSystemDir = "linux_" + (t64Bit ? "64" : "32");
        } else if (OS_NAME.startsWith("windows")) {
            tSystemDir = "windows_" + (t64Bit ? "64" : "32");
        } else if (OS_NAME.startsWith("Mac")) {
            tSystemDir = "osx_64";
        }

        // Prepare temporary file
        try {
            File tTempFile = File.createTempFile("MandelExplorer", "native");
            tTempFile.deleteOnExit();
            String tLibResource = "/natives/" + tSystemDir + "/" + tSystemLibName;
            try (InputStream tLibStream = NativeLoader.class.getResourceAsStream(tLibResource);
                 OutputStream tOutputStream = new FileOutputStream(tTempFile)) {

                if ( tLibStream==null ) {
                    System.out.println(tLibResource + " not found");
                    return;
                }

                byte[] tBuffer = new byte[1024];
                int tReadBytes;
                while ((tReadBytes = tLibStream.read(tBuffer)) != -1) {
                    tOutputStream.write(tBuffer, 0, tReadBytes);
                }
            }

            System.load(tTempFile.getAbsolutePath());
        } catch (IOException pE) {
            pE.printStackTrace();
            return;
        }
    }
}
