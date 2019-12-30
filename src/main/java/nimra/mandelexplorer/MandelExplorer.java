package nimra.mandelexplorer;

import com.aparapi.Range;
import com.aparapi.device.Device;
import com.aparapi.internal.kernel.KernelManager;

import javax.swing.*;
import java.awt.BorderLayout;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Point;
import java.awt.Toolkit;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseWheelEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.IntStream;

/**
 *
 */
public class MandelExplorer {

    private final MandelParams mandelParams = new MandelParams();

    /** User selected zoom-in point on the Mandelbrot view. */
    private Point lastMouseDrag;
    private double toX = mandelParams.getX();
    private double toY = mandelParams.getY();
    private double toScale = mandelParams.getScale();

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

        explorerConfigPanel.setChangeListener(e -> {
            render();
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
                tMaxIterations = (int)(100 * Math.pow(mandelParams.getScale(), -0.4));
            }
            mandelParams.setMaxIterations(tMaxIterations);
            mandelParams.setEscapeRadius(explorerConfigPanel.getEscapeRadius());

            explorerConfigPanel.setData(mandelParams);

            if (viewer.getSize().width != getImageWidth() ||
                viewer.getSize().height != getImageHeight()) {
                setSize(viewer.getSize().width, viewer.getSize().height);
            }

            KernelManager.setKernelManager(new KernelManager() {

                @Override
                protected List<Device.TYPE> getPreferredDeviceTypes() {
                    return Collections.singletonList(explorerConfigPanel.getDeviceType());
                }
            });

            MandelKernel tMandelKernel = getMandelKernel();
            explorerConfigPanel.setAlgoInfo(tMandelKernel);
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

        final int[] imageRgb = ((DataBufferInt)image.getRaster().getDataBuffer()).getData();

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

    MandelKernel doubleMandel;
    MandelKernel floatMandel;
    MandelKernel ddMantel;

    private void prepareMandelKernel() {
        if (doubleMandel != null) {
            doubleMandel.dispose();
        }
        if (floatMandel != null) {
            floatMandel.dispose();
        }

        doubleMandel = new DoubleMandelImpl(getImageWidth(), getImageHeight());
        floatMandel = new FloatMandelImpl(getImageWidth(), getImageHeight());
        ddMantel = new DDMandelImpl(getImageWidth(), getImageHeight());

        explorerConfigPanel.setAlgorithms(floatMandel, doubleMandel, ddMantel);
    }

    private MandelKernel getMandelKernel() {
        MandelKernel tMandelKernel = explorerConfigPanel.getSelectedAlgorithm();

        // auto choose kernel
        if (tMandelKernel == null) {
            final double tMinPixelSize = mandelParams.getScale() / Math.max(getImageHeight(), getImageWidth());
            if (tMinPixelSize < 1E-7) {
                tMandelKernel = doubleMandel;
            } else {
                tMandelKernel = floatMandel;
            }
        }

        return tMandelKernel;
    }

    private void prepareMouse() {

        viewer.addMouseWheelListener(new MouseAdapter() {
            @Override
            public void mouseWheelMoved(final MouseWheelEvent e) {
                toScale += toScale * 0.005f * e.getPreciseWheelRotation() * e.getScrollAmount() * explorerConfigPanel.getZoomSpeed();
                render();
            }
        });

        lastMouseDrag = null;
        viewer.addMouseMotionListener(new MouseAdapter() {
            @Override
            public void mouseDragged(final MouseEvent e) {
                if (!lastMouseDrag.equals(e.getPoint())) {
                    toX = toX + ((lastMouseDrag.x - e.getX()) * mandelParams.getScale()) / getImageWidth();
                    toY = toY + ((lastMouseDrag.y - e.getY()) * mandelParams.getScale()) / getImageHeight();

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
                    final double tScaleX = mandelParams.getScale() * (getImageWidth() / (double)getImageHeight());
                    final double tScaleY = mandelParams.getScale();

                    final int tX = e.getX();
                    final int tY = e.getY();

                    toX = (((tX * tScaleX) - ((tScaleX / 2) * getImageWidth())) / getImageWidth()) + mandelParams.getX();
                    toY = (((tY * tScaleY) - ((tScaleY / 2) * getImageHeight())) / getImageHeight()) + mandelParams.getY();

                    if (e.isControlDown()) {
                        toScale = toScale * explorerConfigPanel.getZoomSpeed();
                    } else {
                        toScale = toScale / explorerConfigPanel.getZoomSpeed();
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

        prepareMandelKernel();
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
