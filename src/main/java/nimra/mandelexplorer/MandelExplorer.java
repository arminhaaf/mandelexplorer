/**
 * Copyright (c) 2016 - 2018 Syncleus, Inc.
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * <p>
 * This product currently only contains code developed by authors
 * of specific components, as identified by the source code files.
 * <p>
 * Since product implements StAX API, it has dependencies to StAX API
 * classes.
 * <p>
 * For additional credits (generally to people who reported problems)
 * see CREDITS file.
 * <p>
 * This product currently only contains code developed by authors
 * of specific components, as identified by the source code files.
 * <p>
 * Since product implements StAX API, it has dependencies to StAX API
 * classes.
 * <p>
 * For additional credits (generally to people who reported problems)
 * see CREDITS file.
 */
/**
 * This product currently only contains code developed by authors
 * of specific components, as identified by the source code files.
 *
 * Since product implements StAX API, it has dependencies to StAX API
 * classes.
 *
 * For additional credits (generally to people who reported problems)
 * see CREDITS file.
 */
/*
Copyright (c) 2010-2011, Advanced Micro Devices, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following
disclaimer. 

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided with the distribution. 

Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

If you use the software (in whole or in part), you shall adhere to all applicable U.S., European, and other export
laws, including but not limited to the U.S. Export Administration Regulations ("EAR"), (15 C.F.R. Sections 730 through
774), and E.U. Council Regulation (EC) No 1334/2000 of 22 June 2000.  Further, pursuant to Section 740.6 of the EAR,
you hereby certify that, except pursuant to a license granted by the United States Department of Commerce Bureau of 
Industry and Security or as otherwise permitted pursuant to a License Exception under the U.S. Export Administration 
Regulations ("EAR"), you will not (1) export, re-export or release to a national of a country in Country Groups D:1,
E:1 or E:2 any restricted technology, software, or source code you receive hereunder, or (2) export to Country Groups
D:1, E:1 or E:2 the direct product of such technology or software, if such foreign produced direct product is subject
to national security controls as identified on the Commerce Control List (currently found in Supplement 1 to Part 774
of EAR).  For the most current Country Group listings, or for additional information about the EAR or your obligations
under those regulations, please refer to the U.S. Bureau of Industry and Security's website at http://www.bis.doc.gov/. 

*/

package nimra.mandelexplorer;

import com.aparapi.Range;
import com.aparapi.device.Device;
import com.aparapi.internal.kernel.KernelManager;

import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JSplitPane;
import java.awt.BorderLayout;
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
                tMaxIterations = (int)(200 * Math.pow(mandelParams.getScale(), -0.8));
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

    private void prepareMandelKernel() {
        if (doubleMandel != null) {
            doubleMandel.dispose();
        }
        if (floatMandel != null) {
            floatMandel.dispose();
        }

        doubleMandel = new DoubleMandelImpl(getImageWidth(), getImageHeight());
        floatMandel = new FloatMandelImpl(getImageWidth(), getImageHeight());

        explorerConfigPanel.setAlgorithms(floatMandel, doubleMandel);
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
