package nimra.mandelexplorer;

import com.aparapi.device.Device;
import com.aparapi.device.OpenCLDevice;

import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JSplitPane;
import javax.swing.WindowConstants;
import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Point;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseWheelEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.math.BigDecimal;
import java.math.MathContext;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.IntStream;

/**
 * Created: 12.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class FractalExplorer {

    private final MandelParams mandelParams = new MandelParams();

    private MandelResult currentMandelResult;

    //private MandelImpl currentMandelImpl = new FractalDMandelImpl(new JavaFractalDFunction("tResult.set(z).sqr().mul(z).add(c);", "ZHoch3"));

    private final CalcStatistics calcStatistics = new CalcStatistics();

    /**
     * User selected zoom-in point on the Mandelbrot view.
     */
    private Point lastMouseDrag;
    private BigDecimal toX = mandelParams.getX();
    private BigDecimal toY = mandelParams.getY();
    private BigDecimal toScale = mandelParams.getScale();

    private final AtomicBoolean doorBell = new AtomicBoolean(true);

    private MandelConfigPanel explorerConfigPanel = new MandelConfigPanel();

    private BufferedImage image;

    private int coarseFactor = 10;

    private List<ComputeDevice> enabledDevices = new ArrayList<>();


    // Draw  image
    private final JComponent viewer = new JComponent() {
        @Override
        public void paintComponent(Graphics g) {
            g.drawImage(image, 0, 0, getImageWidth(), viewer.getHeight(), this);
        }
    };

    private final JSplitPane panel = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, false, explorerConfigPanel.getComponent(), viewer);

    public FractalExplorer() {
        setSize(1024, 1024);

        panel.setOneTouchExpandable(true);

        prepareMouse();

        explorerConfigPanel.setChangeListener(e -> render());

        explorerConfigPanel.setPaletteChangeListener(e -> {
            paint();
            viewer.repaint();
        });

        viewer.addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(final ComponentEvent e) {
                render();
            }
        });

        viewer.setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));

        initPaintThread();

        initComputeDevices();
    }

    private void initComputeDevices() {
        enabledDevices.add(ComputeDevice.CPU);
        for (OpenCLDevice tOpenCLGPU : OpenCLDevice.listDevices(Device.TYPE.GPU)) {
            enabledDevices.add(new ComputeDevice("OpenCL " + tOpenCLGPU.getName(), tOpenCLGPU));
            return;
        }
    }

    protected void initPaintThread() {
        final Thread paintThread = new Thread(() -> {
            while (true) {

                paintDoorBell.set(false);

                try {
                    doPaint();
                } catch (Exception ex) {
                    ex.printStackTrace();
                }

                // Wait for the user to click somewhere
                synchronized (paintDoorBell) {
                    if (!paintDoorBell.get()) {
                        try {
                            paintDoorBell.wait();
                        } catch (final InterruptedException ie) {
                            ie.getStackTrace();
                        }
                    }
                }

            }
        });
        paintThread.start();
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
                    final BigDecimal tScaleX = mandelParams.getScale().multiply(new BigDecimal(getImageWidth())).divide(new BigDecimal(getImageHeight()), MathContext.DECIMAL128);
                    // final double tScaleY = mandelParams.getScale_double();
                    final BigDecimal tScaleY = mandelParams.getScale();

                    //final double xStart =  mandelParams.getX_Double() - tScaleX / 2.0;
                    final BigDecimal xStart = mandelParams.getX().subtract(tScaleX.multiply(BD_0_5));
                    //final double yStart =  mandelParams.getY_Double() - tScaleY / 2.0;
                    final BigDecimal yStart = mandelParams.getY().subtract(tScaleY.multiply(BD_0_5));

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


    private void render() {
        synchronized (doorBell) {
            doorBell.set(true);
            doorBell.notify();
        }
    }

    private void setSize(int pWidth, int pHeight) {
        image = new BufferedImage(Math.max(1, pWidth), Math.max(1, pHeight), BufferedImage.TYPE_INT_RGB);
        // Set the size of JComponent which displays Mandelbrot image
        viewer.setPreferredSize(new Dimension(getImageWidth(), getImageHeight()));

        currentMandelResult = new MandelResult(getImageWidth(), getImageHeight());
    }

    private int getImageWidth() {
        return image != null ? image.getWidth() : 0;
    }

    private int getImageHeight() {
        return image != null ? image.getHeight() : 0;
    }

    private Component getComponent() {
        return panel;
    }

    public void calcCoarse(MandelImpl pMandelImpl) {
        long tStart = System.currentTimeMillis();
        final int tCoarseWidth = getImageWidth() / coarseFactor;
        final int tCoarseHeight = getImageHeight() / coarseFactor;

        if (tCoarseHeight <= 0 || tCoarseWidth <= 0) {
            return;
        }

        final MandelResult tCoarseResult = new MandelResult(tCoarseWidth, tCoarseHeight);

        mandelParams.setCalcMode(explorerConfigPanel.getMode());

        computation.compute(doorBell, pMandelImpl, enabledDevices, mandelParams, tCoarseResult, 1);

        // daten kopieren                                                                ^
        for (int y = 0; y < getImageHeight(); y++) {
            for (int x = 0; x < getImageWidth(); x++) {
                int i = x + y * getImageWidth();
                final int tCoarseIndex = Math.min(x / coarseFactor + (y / coarseFactor) * tCoarseWidth, tCoarseResult.iters.length - 1);
                currentMandelResult.iters[i] = tCoarseResult.iters[tCoarseIndex];
                currentMandelResult.lastValuesR[i] = tCoarseResult.lastValuesR[tCoarseIndex];
                currentMandelResult.lastValuesI[i] = tCoarseResult.lastValuesI[tCoarseIndex];
                currentMandelResult.distancesR[i] = tCoarseResult.distancesR[tCoarseIndex];
                currentMandelResult.distancesI[i] = tCoarseResult.distancesI[tCoarseIndex];
            }
        }
        calcStatistics.addCalcCoarse(tStart);
    }

    final MultiDeviceComputation computation = new MultiDeviceComputation(pTile -> paint());

    public void calcTiles(MandelImpl pMandelImpl) {
        // Tiles for width
        int tTileCount = explorerConfigPanel.getTiles();

        if (tTileCount <= 0) {
            //  TODO auto calc tile count
            tTileCount = 10;
        }

        long tStartMillis = System.currentTimeMillis();
        computation.compute(doorBell, pMandelImpl, enabledDevices, mandelParams, currentMandelResult, tTileCount);
        calcStatistics.addCalc(tStartMillis);
    }

    private final AtomicBoolean paintDoorBell = new AtomicBoolean(false);

    private void paint() {
        synchronized (paintDoorBell) {
            paintDoorBell.set(true);
            paintDoorBell.notify();
        }
    }


    private void doPaint() {
        long tStartMillis = System.currentTimeMillis();

        final int[] imageRgb = ((DataBufferInt)image.getRaster().getDataBuffer()).getData();

        final PaletteMapper tPaletteMapper = explorerConfigPanel.getPaletteMapper().clone();
        final MandelResult tCurrentMandelResult = currentMandelResult;
        final int[] tIters = tCurrentMandelResult.iters;
        final double[] tLastR = tCurrentMandelResult.lastValuesR;
        final double[] tLastI = tCurrentMandelResult.lastValuesI;
        final double[] tDistancesR = tCurrentMandelResult.distancesR;
        final double[] tDistancesI = tCurrentMandelResult.distancesI;
        tPaletteMapper.init(mandelParams);

        // parallelize this -> some palettemappers needs a lot power -> DistanceLight
        final IntStream tPrepareStream = IntStream.range(0, imageRgb.length);
        tPrepareStream.parallel().forEach(i -> tPaletteMapper.prepare(tIters[i], tLastR[i], tLastI[i], tDistancesR[i], tDistancesI[i]));
        tPaletteMapper.startMap();
        final IntStream tMapStream = IntStream.range(0, imageRgb.length);
        tMapStream.parallel().forEach(i -> imageRgb[i] = tPaletteMapper.map(tIters[i], tLastR[i], tLastI[i], tDistancesR[i], tDistancesI[i]));
        calcStatistics.addPaintDuration(tStartMillis);

        tStartMillis = System.currentTimeMillis();
        viewer.repaint();
        calcStatistics.addRepaintDuration(tStartMillis);

    }

    private void start() {

        while (true) {
            calcStatistics.reset();

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
                tMaxIterations = (int)(100 * Math.pow(mandelParams.getScale_double(), -0.4));
            }
            mandelParams.setMaxIterations(tMaxIterations);
            mandelParams.setEscapeRadius(explorerConfigPanel.getEscapeRadius());
            mandelParams.setJuliaCr(explorerConfigPanel.getJuliaCr());
            mandelParams.setJuliaCi(explorerConfigPanel.getJuliaCi());
            mandelParams.setCalcMode(explorerConfigPanel.getMode());

            explorerConfigPanel.setData(mandelParams);

            if (viewer.getSize().width != getImageWidth() ||
                viewer.getSize().height != getImageHeight()) {
                setSize(viewer.getSize().width, viewer.getSize().height);
            }

            final MandelImpl tMandelImpl = explorerConfigPanel.getMandelImpl();

            calcCoarse(tMandelImpl);
            paint();

            try {
                viewer.setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
                calcTiles(tMandelImpl);
                explorerConfigPanel.setRenderMillis(calcStatistics.calcMillis, calcStatistics.paintMillis);
            } catch (Exception ex) {
                ex.printStackTrace();
                JOptionPane.showMessageDialog(null, ex.getMessage());
            } finally {
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


    public static void main(String[] args) {
        final JFrame frame = new JFrame("Mandel-Explorer");

        final FractalExplorer tFractalExplorer = new FractalExplorer();


        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.add(tFractalExplorer.getComponent(), BorderLayout.CENTER);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        try {
            tFractalExplorer.start();
        } catch (Exception ex) {
            ex.printStackTrace();
            JOptionPane.showMessageDialog(null, ex.getMessage());
            System.exit(-1);
        }
    }

    static class CalcStatistics {
        long calcCoarseMillis;

        long calcMillis;

        long paintMillis;

        long repaintMillis;

        public void addCalcCoarse(long pStartMillis) {
            calcCoarseMillis += System.currentTimeMillis() - pStartMillis;
        }

        public void addCalc(long pStartMillis) {
            calcMillis += System.currentTimeMillis() - pStartMillis;
        }

        public void addPaintDuration(long pStartMillis) {
            paintMillis += System.currentTimeMillis() - pStartMillis;
        }

        public void reset() {
            calcMillis = paintMillis = 0;
        }

        public void addRepaintDuration(final long pStartMillis) {
            repaintMillis += System.currentTimeMillis() - pStartMillis;
        }
    }


}
