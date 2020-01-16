package nimra.mandelexplorer;

import nimra.mandelexplorer.palette.DefaultPaletteMapper;

import javax.swing.JComponent;
import javax.swing.JOptionPane;
import javax.swing.event.ChangeListener;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.math.BigDecimal;
import java.math.MathContext;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.IntStream;

/**
 * Created: 14.01.20   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class JuliaChooser {

    private final BigDecimal BD_0_5 = new BigDecimal("0.5");

    private ChangeListener changeListener;

    private BufferedImage image;

    private final AtomicBoolean doorBell = new AtomicBoolean(true);

    private MandelImpl mandelImpl = new StreamParallelDoubleMandelImpl();

    private PaletteMapper paletteMapper = new DefaultPaletteMapper();

    private MandelParams mandelParams = new MandelParams();

    private boolean enabled = false;

    // Draw  image
    private final JComponent viewer = new JComponent() {
        @Override
        public void paintComponent(Graphics g) {
            if (image != null) {
                g.drawImage(image, 0, 0, getImageWidth(), getImageHeight(), this);

                // paint simple crosshair
                int tCrossHairSize = 20;

                final BigDecimal tScaleX = mandelParams.getScale().multiply(new BigDecimal(getImageWidth())).divide(new BigDecimal(getImageHeight()), MathContext.DECIMAL128);
                final BigDecimal tScaleY = mandelParams.getScale();

                //final double xStart =  mandelParams.getX_Double() - tScaleX / 2.0;
                final BigDecimal xStart = mandelParams.getX().subtract(tScaleX.multiply(BD_0_5));
                //final double yStart =  mandelParams.getY_Double() - tScaleY / 2.0;
                final BigDecimal yStart = mandelParams.getY().subtract(tScaleY.multiply(BD_0_5));

                int tCenterX = cr.subtract(xStart).multiply(new BigDecimal(getImageWidth())).divide(tScaleX, MathContext.DECIMAL128).intValue();
                int tCenterY = ci.subtract(yStart).multiply(new BigDecimal(getImageHeight())).divide(tScaleY, MathContext.DECIMAL128).intValue();

                g.setColor(Color.WHITE);
                g.drawLine(tCenterX, tCenterY - tCrossHairSize / 2, tCenterX, tCenterY + tCrossHairSize / 2);
                g.drawLine(tCenterX - tCrossHairSize / 2, tCenterY, tCenterX + tCrossHairSize / 2, tCenterY);
            }
        }
    };

    public void setMandelImpl(MandelImpl pMandelImpl) {
        if (pMandelImpl != mandelImpl) {
            mandelImpl = pMandelImpl;
            mandelParams = mandelImpl.getHomeParams();
            paletteMapper = mandelImpl.getDefaultPaletteMapper();
            render();
        }
    }

    protected int getImageHeight() {
        return image.getHeight();
    }

    protected int getImageWidth() {
        return image.getWidth();
    }

    private BigDecimal cr = new BigDecimal(0);
    private BigDecimal ci = new BigDecimal(0);

    public JuliaChooser(final ChangeListener pChangeListener) {
        changeListener = pChangeListener;

        setSize(100, 100);

        viewer.addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(final ComponentEvent e) {
                if (viewer.getSize().width != getImageWidth() ||
                    viewer.getSize().height != getImageHeight()) {
                    setSize(viewer.getSize().width, viewer.getSize().height);
                    render();
                }
            }
        });

        new Thread(this::start).start();

        prepareMouse();
    }

    public BigDecimal getCr() {
        return cr;
    }

    public BigDecimal getCi() {
        return ci;
    }


    private void prepareMouse() {

        viewer.addMouseMotionListener(new MouseAdapter() {
            @Override
            public void mouseDragged(final MouseEvent e) {
                setFromMouse(e);
            }
        });
        // Mouse listener which reads the user clicked zoom-in point on the Mandelbrot view
        viewer.addMouseListener(new MouseAdapter() {

            @Override
            public void mouseClicked(MouseEvent e) {
                if (e.getClickCount() == 1) {
                    setFromMouse(e);
                }
            }
        });
    }

    private void setFromMouse(MouseEvent e) {
        if (enabled) {
            // final double tScaleX = mandelParams.getScale_double() * (getImageWidth() / (double) getImageHeight());
            final BigDecimal tScaleX = mandelParams.getScale().multiply(new BigDecimal(getImageWidth())).divide(new BigDecimal(getImageHeight()), MathContext.DECIMAL128);
            // final double tScaleY = mandelParams.getScale_double();
            final BigDecimal tScaleY = mandelParams.getScale();

            //final double xStart =  mandelParams.getX_Double() - tScaleX / 2.0;
            final BigDecimal xStart = mandelParams.getX().subtract(tScaleX.multiply(BD_0_5));
            //final double yStart =  mandelParams.getY_Double() - tScaleY / 2.0;
            final BigDecimal yStart = mandelParams.getY().subtract(tScaleY.multiply(BD_0_5));

            cr = xStart.add(tScaleX.multiply(new BigDecimal(e.getX())).divide(new BigDecimal(getImageWidth()), MathContext.DECIMAL128));
            ci = yStart.add(tScaleY.multiply(new BigDecimal(e.getY())).divide(new BigDecimal(getImageHeight()), MathContext.DECIMAL128));
            changeListener.stateChanged(null);
            viewer.repaint();
        }
    }

    private void setSize(int pWidth, int pHeight) {
        image = new BufferedImage(Math.max(1, pWidth), Math.max(1, pHeight), BufferedImage.TYPE_INT_RGB);
        // Set the size of JComponent which displays Mandelbrot image
        viewer.setPreferredSize(new Dimension(getImageWidth(), getImageHeight()));
    }

    private void render() {
        synchronized (doorBell) {
            doorBell.set(true);
            doorBell.notify();
        }
    }

    private void start() {
        while (true) {
            doorBell.set(false);
            try {
                viewer.setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));

                if (viewer.getSize().width != getImageWidth() ||
                    viewer.getSize().height != getImageHeight()) {
                    setSize(viewer.getSize().width, viewer.getSize().height);
                }

                calc();
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

    private void calc() {
        final MandelResult tResult = new MandelResult(getImageWidth(), getImageHeight());

        MandelImpl tAlgo = mandelImpl.isThreadSafe() ? mandelImpl : mandelImpl.copy();

        tAlgo.mandel(mandelParams,
                     tResult, new Tile(0, 0, getImageWidth(), getImageHeight())
        );

        final int[] imageRgb = ((DataBufferInt)image.getRaster().getDataBuffer()).getData();

        final PaletteMapper tPaletteMapper = paletteMapper.clone();

        tPaletteMapper.init(mandelParams);

        // parallelize this -> some palettemappers needs a lot power -> DistanceLight
        final IntStream tPrepareStream = IntStream.range(0, imageRgb.length);
        tPrepareStream.parallel().forEach(i -> tPaletteMapper.prepare(tResult.iters[i], tResult.lastValuesR[i], tResult.lastValuesI[i],
                                                                      tResult.distancesR[i], tResult.distancesI[i]));
        tPaletteMapper.startMap();
        final IntStream tMapStream = IntStream.range(0, imageRgb.length);
        tMapStream.parallel().forEach(i -> imageRgb[i] = tPaletteMapper.map(tResult.iters[i], tResult.lastValuesR[i], tResult.lastValuesI[i],
                                                                            tResult.distancesR[i], tResult.distancesI[i]));
        viewer.repaint();
    }

    public JComponent getComponent() {
        return viewer;
    }

    public void setEnabled(final boolean pEnabled) {
        enabled = pEnabled;
    }
}
