package nimra.mandelexplorer;

import com.jgoodies.forms.layout.CellConstraints;
import com.jgoodies.forms.layout.FormLayout;
import nimra.mandelexplorer.palette.DefaultPaletteMapper;

import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.JWindow;
import javax.swing.event.ChangeListener;
import java.awt.BorderLayout;
import java.awt.Color;
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

    private JWindow window;

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
    private JPanel mainPanel;
    private JTextField crTextField;
    private JTextField ciTextField;

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

        mainPanel.add(viewer);

        new Thread(this::start).start();

        prepareMouse();

        updateTextField();

        crTextField.addActionListener(e -> {
            cr = new BigDecimal(crTextField.getText());
            changed();
        });
        ciTextField.addActionListener(e -> {
            ci = new BigDecimal(ciTextField.getText());
            changed();
        });
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
        if (enabled && !isMoveEvent(e)) {
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
            changed();
        }
    }

    private boolean isMoveEvent(MouseEvent e) {
        return e.isControlDown();
    }

    private void changed() {
        updateTextField();
        changeListener.stateChanged(null);
        viewer.repaint();
    }

    private void updateTextField() {
        crTextField.setText(cr.toString());
        crTextField.setCaretPosition(0);
        ciTextField.setText(ci.toString());
        ciTextField.setCaretPosition(0);
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

    private Point mouseClickPoint;

    private void addEventsForDragging() {
        // Here is the code does moving
        viewer.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                if (isMoveEvent(e)) {
                    mouseClickPoint = e.getPoint();
                }
            }
        });
        viewer.addMouseMotionListener(new MouseAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                if (isMoveEvent(e)) {
                    Point tNewLocation = e.getLocationOnScreen();
                    tNewLocation.translate(-mouseClickPoint.x, -mouseClickPoint.y);
                    window.setLocation(tNewLocation);
                }
            }
        });
        viewer.addMouseWheelListener(new MouseAdapter() {
            @Override
            public void mouseWheelMoved(final MouseWheelEvent e) {
                if (isMoveEvent(e)) {
                    final double tZoomFactor = e.getWheelRotation() * e.getScrollAmount() * 0.01;
                    window.setSize(window.getWidth() + (int)(window.getWidth() * tZoomFactor), window.getHeight() + (int)(window.getHeight() * tZoomFactor));
                }
            }
        });
    }

    public void showInWindow(JComponent pOwner) {

        if (window == null) {
            window = new JWindow(JOptionPane.getFrameForComponent(pOwner));
            window.add(getComponent());
            window.setSize(200, 200);
            window.setVisible(true);
            final Point tLocationOnScreen = pOwner.getLocationOnScreen();
            window.setLocation((tLocationOnScreen.x + pOwner.getWidth()) - window.getWidth(),
                               (tLocationOnScreen.y + pOwner.getHeight()) - window.getHeight());
            addEventsForDragging();
        }
        window.setVisible(true);
        setEnabled(true);
    }

    public void hide() {
        if (window != null) {
            window.setVisible(false);
            setEnabled(false);
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

        for (ComputeDevice tComputeDevice : new ArrayList<>(ComputeDevice.DEVICES)) {
            if (tComputeDevice.isEnabled() && mandelImpl.supports(tComputeDevice)) {
                final MandelImpl tAlgo = mandelImpl.isThreadSafe() ? mandelImpl : mandelImpl.copy();

                tAlgo.mandel(tComputeDevice, mandelParams,
                             tResult, new Tile(0, 0, getImageWidth(), getImageHeight()));
                break;
            }
        }

        final int[] imageRgb = ((DataBufferInt)image.getRaster().getDataBuffer()).getData();

        if (imageRgb.length != tResult.iters.length) {
            // inconsistent view sizes
            return;
        }

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
        return mainPanel;
    }

    public void setEnabled(final boolean pEnabled) {
        enabled = pEnabled;
    }

    {
// GUI initializer generated by IntelliJ IDEA GUI Designer
// >>> IMPORTANT!! <<<
// DO NOT EDIT OR ADD ANY CODE HERE!
        $$$setupUI$$$();
    }

    /**
     * Method generated by IntelliJ IDEA GUI Designer
     * >>> IMPORTANT!! <<<
     * DO NOT edit this method OR call it in your code!
     *
     * @noinspection ALL
     */
    private void $$$setupUI$$$() {
        mainPanel = new JPanel();
        mainPanel.setLayout(new BorderLayout(0, 0));
        mainPanel.setOpaque(false);
        final JPanel panel1 = new JPanel();
        panel1.setLayout(new FormLayout("fill:d:noGrow,left:4dlu:noGrow,fill:d:grow,left:4dlu:noGrow,fill:max(d;4px):noGrow,left:4dlu:noGrow,fill:max(d;4px):grow", "center:d:grow,top:4dlu:noGrow,center:max(d;4px):noGrow"));
        panel1.setOpaque(false);
        mainPanel.add(panel1, BorderLayout.SOUTH);
        final JLabel label1 = new JLabel();
        label1.setText("CR");
        CellConstraints cc = new CellConstraints();
        panel1.add(label1, cc.xy(1, 3));
        crTextField = new JTextField();
        panel1.add(crTextField, cc.xy(3, 3, CellConstraints.FILL, CellConstraints.DEFAULT));
        final JLabel label2 = new JLabel();
        label2.setText("CI");
        panel1.add(label2, cc.xy(5, 3));
        ciTextField = new JTextField();
        panel1.add(ciTextField, cc.xy(7, 3, CellConstraints.FILL, CellConstraints.DEFAULT));
    }

    /**
     * @noinspection ALL
     */
    public JComponent $$$getRootComponent$$$() {
        return mainPanel;
    }

    public void set(final BigDecimal pJuliaCr, final BigDecimal pJuliaCi) {
        cr = pJuliaCr;
        ci = pJuliaCi;
        changed();
    }
}
