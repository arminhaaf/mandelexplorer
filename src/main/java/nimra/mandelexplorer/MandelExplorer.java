package nimra.mandelexplorer;

import com.aparapi.Range;
import com.aparapi.device.Device;
import com.aparapi.internal.kernel.KernelManager;

import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JSplitPane;
import javax.swing.Timer;
import javax.swing.WindowConstants;
import java.awt.BorderLayout;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Point;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseWheelEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.math.BigDecimal;
import java.math.MathContext;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.IntStream;

/**
 *
 */
public class MandelExplorer {

    private final MandelParams mandelParams = new MandelParams();

    /**
     * User selected zoom-in point on the Mandelbrot view.
     */
    private Point lastMouseDrag;
    private BigDecimal toX = mandelParams.getX();
    private BigDecimal toY = mandelParams.getY();
    private BigDecimal toScale = mandelParams.getScale();

    private MandelConfigPanel explorerConfigPanel = new MandelConfigPanel();

    private final AtomicBoolean doorBell = new AtomicBoolean(true);

    private Range range;

    /**
     * Image for Mandelbrot view.
     */
    private BufferedImage image;

    // Draw  image
    private final JComponent viewer = new JComponent() {
        @Override
        public void paintComponent(Graphics g) {
            g.drawImage(image, 0, 0, getImageWidth(), getImageHeight(), this);
        }
    };

    private final JSplitPane panel = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, false, explorerConfigPanel.getComponent(), viewer);

    public MandelExplorer() {
        setSize(1024, 1024);

        panel.setOneTouchExpandable(true);

        prepareMouse();

        explorerConfigPanel.setChangeListener(e -> render());

        explorerConfigPanel.setPaletteChangeListener(e -> {
            paint(currentMandelKernelAlgo.getMandelKernel());
            viewer.repaint();
        });

        viewer.addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(final ComponentEvent e) {
                render();
            }
        });

        viewer.setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
    }

    private void render() {
        synchronized (doorBell) {
            doorBell.set(true);
            doorBell.notify();
        }
    }


    public JComponent getComponent() {
        return panel;
    }

    private MandelAlgo currentMandelKernelAlgo;

    private void start() {

        while (true) {

            doorBell.set(false);

            if (explorerConfigPanel.getX() != null) {
                toX = explorerConfigPanel.getX();
            }
            if (explorerConfigPanel.getY() != null) {
                toY = explorerConfigPanel.getY();
            }
            if (explorerConfigPanel.getScale() != null) {
                toScale = explorerConfigPanel.getScale();
            }
            mandelParams.setX(toX);
            mandelParams.setY(toY);
            mandelParams.setScale(toScale);

            int tMaxIterations = explorerConfigPanel.getMaxIterations();

            if (tMaxIterations < 0) {
                tMaxIterations = (int) (100 * Math.pow(mandelParams.getScale_double(), -0.4));
            }
            mandelParams.setMaxIterations(tMaxIterations);
            mandelParams.setEscapeRadius(explorerConfigPanel.getEscapeRadius());

            explorerConfigPanel.setData(mandelParams);

            if (viewer.getSize().width != getImageWidth() ||
                    viewer.getSize().height != getImageHeight()) {
                setSize(viewer.getSize().width, viewer.getSize().height);
            }

            Timer tInfoTimer = new Timer(10, new ActionListener() {
                @Override
                public void actionPerformed(final ActionEvent e) {
//                    if ( currentMandelKernel.isExecuting()) {
//                        System.out.println(currentMandelKernel.getCurrentPass());
//                    } else {
//                        System.out.println("done");
//                    }
                }
            });

            currentMandelKernelAlgo = getMandelAlgo();

            final MandelKernel tMandelKernel = currentMandelKernelAlgo.getMandelKernel();
            tMandelKernel.setCalcDistance(explorerConfigPanel.calcDistance());

            final LinkedHashSet<Device> tPreferredDevices = new LinkedHashSet<>();
            tPreferredDevices.add(explorerConfigPanel.getDevice());
            KernelManager.instance().setPreferredDevices(tMandelKernel, tPreferredDevices);

            explorerConfigPanel.setAlgoInfo(currentMandelKernelAlgo);

            tInfoTimer.start();
            try {
                viewer.setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
                long tStartCalcMillis = System.currentTimeMillis();
                calc(tMandelKernel);
                long tCalcMillis = System.currentTimeMillis() - tStartCalcMillis;
                long tStartColorMillis = System.currentTimeMillis();
                paint(tMandelKernel);
                long tColorMillis = System.currentTimeMillis() - tStartColorMillis;
                explorerConfigPanel.setRenderMillis(tCalcMillis, tColorMillis);
                viewer.repaint();
            } catch (Exception ex) {
                ex.printStackTrace();
                JOptionPane.showMessageDialog(null, ex.getMessage());
            } finally {
                tInfoTimer.stop();
                viewer.setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
            }

            // Wait for the user to click somewhere
            synchronized (doorBell) {
                if (!doorBell.get()) {
                    try {
                        doorBell.wait();
                    } catch (final InterruptedException ie) {
                        ie.getStackTrace();
                    }
                }
            }

        }
    }

    private void paint(MandelKernel pMandelKernel) {

        final int[] imageRgb = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();

        final PaletteMapper tPaletteMapper = explorerConfigPanel.getPaletteMapper();
        final int[] tIters = pMandelKernel.getIters();
        final double[] tLastR = pMandelKernel.getLastValuesR();
        final double[] tLastI = pMandelKernel.getLastValuesI();
        final double[] tDistancesR = pMandelKernel.getDistancesR();
        final double[] tDistancesI = pMandelKernel.getDistancesI();
        tPaletteMapper.init(mandelParams);

        // parallelize this -> some palettemappers needs a lot power -> DistanceLight
        final IntStream tPrepareStream = IntStream.range(0, imageRgb.length);
        tPrepareStream.parallel().forEach(i -> tPaletteMapper.prepare(tIters[i], tLastR[i], tLastI[i], tDistancesR[i], tDistancesI[i]));
        tPaletteMapper.startMap();
        final IntStream tMapStream = IntStream.range(0, imageRgb.length);
        tMapStream.parallel().forEach(i -> imageRgb[i] = tPaletteMapper.map(tIters[i], tLastR[i], tLastI[i], tDistancesR[i], tDistancesI[i]));
    }

    private void calc(MandelKernel pMandelKernel) {
        pMandelKernel.init(mandelParams);
        pMandelKernel.execute(range);
    }

    private void prepareMandelKernel() {
       explorerConfigPanel.setAlgorithms(
                new MandelAlgo("Float", 1E-4, () -> new FloatMandelKernel(getImageWidth(), getImageHeight())),
                new MandelAlgo("Double", 1E-15,() -> new DoubleMandelImpl(getImageWidth(), getImageHeight())),
                new MandelAlgo("FloatCL", 1E-4, () -> new FloatCLMandelKernel(getImageWidth(), getImageHeight())),
                new MandelAlgo("FFCL", 1E-12, () -> new FFCLMandelKernel(getImageWidth(), getImageHeight())),
                new MandelAlgo("DD", 1E-28, () -> new DDMandelImpl(getImageWidth(), getImageHeight())),
                new MandelAlgo("DDCL",1E-27, () -> new DDCLMandelKernel(getImageWidth(), getImageHeight())),
                new MandelAlgo("FP128CL", 1E-40, () -> new FP128CLMandelKernel(getImageWidth(), getImageHeight()))
                // bugggy
                //new MandelAlgo("QFCL",1E-14, () -> new QFCLMandelKernel(getImageWidth(), getImageHeight()))
                );
    }

    private MandelAlgo getMandelAlgo() {
        MandelAlgo tMandelAlgo = explorerConfigPanel.getSelectedAlgorithm();

        // auto choose kernel
        if (tMandelAlgo == null) {
            boolean tGPU = explorerConfigPanel.getDevice().getType()== Device.TYPE.GPU;
            final double tMinPixelSize = mandelParams.getScale_double() / Math.max(getImageHeight(), getImageWidth());
            final List<MandelAlgo> tAlgos = explorerConfigPanel.getAlgos();
            MandelAlgo tBestMatch = null;
            for (MandelAlgo tAlgo : tAlgos) {
                if ( tAlgo.getPrecision()<tMinPixelSize) {
                    if ( tBestMatch==null ||tBestMatch.getPrecision()<tAlgo.getPrecision()) {
                        tBestMatch = tAlgo;
                    }
                }
            }
            if ( tBestMatch!=null ) {
                return tBestMatch;
            } else {
                return tAlgos.get(0);
            }
        }

        return tMandelAlgo;
    }

    private final BigDecimal BD_0_5 = new BigDecimal("0.5");

    private void prepareMouse() {

        viewer.addMouseWheelListener(new MouseAdapter() {
            @Override
            public void mouseWheelMoved(final MouseWheelEvent e) {
                //toScale += toScale * 0.005f * e.getPreciseWheelRotation() * e.getScrollAmount() * explorerConfigPanel.getZoomSpeed();
                toScale = toScale.add(toScale.divide(new BigDecimal(200), MathContext.DECIMAL128).multiply(new BigDecimal(e.getWheelRotation() * e.getScrollAmount() * explorerConfigPanel.getZoomSpeed())));

                render();
            }
        });

        lastMouseDrag = null;
        viewer.addMouseMotionListener(new MouseAdapter() {
            @Override
            public void mouseDragged(final MouseEvent e) {
                if (!lastMouseDrag.equals(e.getPoint())) {
                    //toX = toX + ((lastMouseDrag.x - e.getX()) * mandelParams.getScale_double()) / getImageWidth();
                    toX = toX.add(new BigDecimal(lastMouseDrag.x - e.getX()).multiply(mandelParams.getScale()).divide(new BigDecimal(getImageWidth()), MathContext.DECIMAL128));
                    //toY = toY + ((lastMouseDrag.y - e.getY()) * mandelParams.getScale_double()) / getImageHeight();
                    toY = toY.add(new BigDecimal(lastMouseDrag.y - e.getY()).multiply(mandelParams.getScale()).divide(new BigDecimal(getImageHeight()), MathContext.DECIMAL128));

                    render();

                    lastMouseDrag = e.getPoint();
                }

            }
        });

        // Mouse listener which reads the user clicked zoom-in point on the Mandelbrot view
        viewer.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(final MouseEvent e) {
                lastMouseDrag = e.getPoint();
            }

            @Override
            public void mouseClicked(MouseEvent e) {
                if (e.getClickCount() == 2) {
                    // final double tScaleX = mandelParams.getScale_double() * (getImageWidth() / (double) getImageHeight());
                    final BigDecimal tScaleX =  mandelParams.getScale().multiply(new BigDecimal(getImageWidth())).divide(new BigDecimal(getImageHeight()), MathContext.DECIMAL128);
                    // final double tScaleY = mandelParams.getScale_double();
                    final BigDecimal tScaleY = mandelParams.getScale();

                    //final double xStart =  mandelParams.getX_Double() - tScaleX / 2.0;
                    final BigDecimal xStart =  mandelParams.getX().subtract(tScaleX.multiply(BD_0_5));
                    //final double yStart =  mandelParams.getY_Double() - tScaleY / 2.0;
                    final BigDecimal yStart =  mandelParams.getY().subtract(tScaleY.multiply(BD_0_5));

//                    double tToX = xStart + e.getX() * tScaleX/(double) getImageWidth();
                    toX = xStart.add(tScaleX.multiply(new BigDecimal(e.getX())).divide(new BigDecimal(getImageWidth()), MathContext.DECIMAL128));
//                    double tToY = yStart + e.getY() * tScaleY/(double) getImageHeight();
                    toY = yStart.add(tScaleY.multiply(new BigDecimal(e.getY())).divide(new BigDecimal(getImageHeight()), MathContext.DECIMAL128));

                    if (e.isControlDown()) {
                        toScale = toScale.multiply(new BigDecimal(explorerConfigPanel.getZoomSpeed()));
                    } else {
                        toScale = toScale.divide(new BigDecimal(explorerConfigPanel.getZoomSpeed()), MathContext.DECIMAL128);
                    }

                    render();
                }
            }
        });

    }

    private void setSize(int pWidth, int pHeight) {
        range = Range.create2D(pWidth, pHeight);

        image = new BufferedImage(getImageWidth(), getImageHeight(), BufferedImage.TYPE_INT_RGB);
        // Set the size of JComponent which displays Mandelbrot image
        viewer.setPreferredSize(new Dimension(getImageWidth(), getImageHeight()));

        cleanupKernels();

        prepareMandelKernel();
    }

    private void cleanupKernels() {
        MandelKernel.disposeAll();
    }

    private int getImageWidth() {
        return range.getGlobalSize(0);
    }

    private int getImageHeight() {
        return range.getGlobalSize(1);
    }


    public static void main(String[] args) {
        Runtime.getRuntime().addShutdownHook(new Thread(MandelKernel::disposeAll));

        final JFrame frame = new JFrame("Mandel-Explorer");

        final MandelExplorer tMandelExplorer = new MandelExplorer();


        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.add(tMandelExplorer.getComponent(), BorderLayout.CENTER);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        try {
            tMandelExplorer.start();
        } catch (Exception ex) {
            ex.printStackTrace();
            JOptionPane.showMessageDialog(null, ex.getMessage());
            System.exit(-1);
        }

    }
}
