package nimra.mandelexplorer;

import com.aparapi.device.Device;
import com.aparapi.device.JavaDevice;
import com.aparapi.device.OpenCLDevice;
import com.jgoodies.forms.layout.CellConstraints;
import com.jgoodies.forms.layout.FormLayout;
import org.apache.commons.lang3.StringUtils;
import org.json.JSONObject;

import javax.swing.DefaultComboBoxModel;
import javax.swing.DefaultListCellRenderer;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSlider;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.event.ChangeListener;
import java.awt.Component;
import java.awt.FlowLayout;
import java.awt.event.ActionListener;
import java.awt.event.FocusAdapter;
import java.awt.event.FocusEvent;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;

/**
 * Created: 27.12.19   by: Armin Haaf
 *
 * @author Armin Haaf
 */
public class MandelConfigPanel {
    private JComboBox<String> maxIterationChooser;
    private JPanel mainPanel;
    private JTextField xTextField;
    private JTextField yTextField;
    private JTextField scaleTextField;
    private JTextField xInfoTextField;
    private JTextField yInfoTextField;
    private JTextField scaleInfoTextField;
    private JTextField maxIterInfoTextField;
    private JComboBox<MandelKernel> algorithmComboBox;
    private JTextField renderMillisTextField;
    private JComboBox<PaletteMapper> paletteComboBox;
    private JTextField algoInfoTextField;
    private JComboBox<Device> deviceComboBox;
    private JComboBox<MandelConfig> configsComboBox;
    private JButton addAsConfigButton;
    private JButton removeSelectedConfig;
    private JTextField escapeRadiusTextField;
    private JTextArea paletteConfigTextArea;
    private JSlider zoomSpeedSlider;
    private JCheckBox calcDistanceCheckBox;

    private ChangeListener changeListener;
    private ChangeListener paletteChangeListener;
    private boolean changeEnabled = true;

    private DefaultComboBoxModel<MandelKernel> algorithmModel = new DefaultComboBoxModel<>();
    private DefaultComboBoxModel<PaletteMapper> paletteModel = new DefaultComboBoxModel<>();

    private final Map<String, MandelConfig> configs = new HashMap<>();

    private Path configFile = Paths.get("mandelConfigs.json");

    public MandelConfigPanel() {
        maxIterationChooser.setEditable(true);
        maxIterationChooser.setModel(new DefaultComboBoxModel<>(new String[]{"Auto", "100", "1000", "10000", "100000"}));


        final ActionListener tActionToChange = e -> changed();
        maxIterationChooser.addActionListener(tActionToChange);

        algorithmComboBox.setModel(algorithmModel);
        algorithmComboBox.addActionListener(tActionToChange);
        algorithmModel.addElement(null);
        algorithmComboBox.setRenderer(new DefaultListCellRenderer() {
            @Override
            public Component getListCellRendererComponent(final JList<?> list, final Object value, final int index, final boolean isSelected, final boolean cellHasFocus) {
                return super.getListCellRendererComponent(list, value != null ? value : "Auto", index, isSelected, cellHasFocus);
            }
        });

        calcDistanceCheckBox.addActionListener(e -> {
            if (getEscapeRadius() < 100 && calcDistanceCheckBox.isSelected()) {
                escapeRadiusTextField.setText("100");
            }
            changed();
        });

        deviceComboBox.setModel(new DefaultComboBoxModel<>(getAvailableDevices().toArray(new Device[0])));
        deviceComboBox.setRenderer(new DefaultListCellRenderer() {
            @Override
            public Component getListCellRendererComponent(final JList<?> list, final Object value, final int index, final boolean isSelected, final boolean cellHasFocus) {
                String tLabel = "JCP";
                if ( value!=null ) {
                    tLabel = ((Device)value).getShortDescription();
                }
                return super.getListCellRendererComponent(list, tLabel, index, isSelected, cellHasFocus);
            }
        });
        deviceComboBox.setSelectedIndex(0);
        deviceComboBox.addActionListener(tActionToChange);

        paletteComboBox.setModel(paletteModel);
        ServiceLoader.load(PaletteMapper.class).forEach(this::addPalette);

        paletteComboBox.addActionListener(e -> {
            paletteConfigTextArea.setText(((PaletteMapper)paletteComboBox.getSelectedItem()).toJson().toString(2));
            paletteChanged();
        });

        xTextField.addActionListener(tActionToChange);
        yTextField.addActionListener(tActionToChange);
        scaleTextField.addActionListener(tActionToChange);
        escapeRadiusTextField.addActionListener(tActionToChange);
        paletteConfigTextArea.addFocusListener(new FocusAdapter() {
            @Override
            public void focusLost(final FocusEvent e) {
                paletteChanged();
            }
        });

        addAsConfigButton.addActionListener(e -> {
            final String tName = JOptionPane.showInputDialog(addAsConfigButton, "Name:");
            if (tName != null) {
                MandelConfig tMandelConfig;
                if (configs.containsKey(tName)) {
                    if (JOptionPane.showConfirmDialog(addAsConfigButton, tName + " already exists! Overwrite ?") != JOptionPane.OK_OPTION) {
                        return;
                    }
                    tMandelConfig = configs.get(tName);
                } else {
                    tMandelConfig = new MandelConfig();
                    configsComboBox.addItem(tMandelConfig);
                    configs.put(tMandelConfig.name, tMandelConfig);
                }
                tMandelConfig.name = tName;
                tMandelConfig.mandelParams = new MandelParams();
                tMandelConfig.mandelParams.setX(new BigDecimal(xInfoTextField.getText()));
                tMandelConfig.mandelParams.setY(new BigDecimal(yInfoTextField.getText()));
                tMandelConfig.mandelParams.setScale(new BigDecimal(scaleInfoTextField.getText()));
                tMandelConfig.mandelParams.setMaxIterations(getMaxIterations());
                tMandelConfig.mandelParams.setEscapeRadius(getEscapeRadius());
                tMandelConfig.palette = ((PaletteMapper)paletteComboBox.getSelectedItem()).getName();
                tMandelConfig.paletteConfigJson = paletteConfigTextArea.getText();
            }
            saveConfigs(configFile);
        });

        removeSelectedConfig.addActionListener(e -> {
            MandelConfig tMandelConfig = (MandelConfig)configsComboBox.getSelectedItem();
            if (tMandelConfig != null) {
                configsComboBox.removeItemAt(configsComboBox.getSelectedIndex());
                configs.remove(tMandelConfig.name);
                saveConfigs(configFile);
            }
        });

        try {
            loadConfig(configFile);
        } catch (IOException ex) {
            ex.printStackTrace();
        }

        configsComboBox.addActionListener(e -> setSelectedConfig());

    }

    private List<Device> getAvailableDevices() {
        List<Device> tAvailableDevices = new ArrayList<>();
        for (Device.TYPE tType : Device.TYPE.values()) {
            tAvailableDevices.addAll(OpenCLDevice.listDevices(tType));
        }
        tAvailableDevices.add(null); // -> JCP
        return tAvailableDevices;
    }

    public double getEscapeRadius() {
        try {
            final double tEscapeRadius = Double.parseDouble(escapeRadiusTextField.getText());
            if (tEscapeRadius > 0) {
                return tEscapeRadius;
            }
        } catch (Exception ex) {
        }

        return 2;
    }

    public boolean calcDistance() {
        return calcDistanceCheckBox.isSelected();
    }

    private void setSelectedConfig() {
        MandelConfig tMandelConfig = (MandelConfig)configsComboBox.getSelectedItem();

        if (tMandelConfig == null) {
            return;
        }

        changeEnabled = false;

        try {
            xTextField.setText(Double.toString(tMandelConfig.mandelParams.getX_Double()));
            yTextField.setText(Double.toString(tMandelConfig.mandelParams.getY_Double()));
            scaleTextField.setText(Double.toString(tMandelConfig.mandelParams.getScale_double()));
            escapeRadiusTextField.setText(Double.toString(tMandelConfig.mandelParams.getEscapeRadius()));
            if (tMandelConfig.mandelParams.getMaxIterations() < 0) {
                maxIterationChooser.setSelectedIndex(0);
            } else {
                maxIterationChooser.setSelectedItem(Integer.toString(tMandelConfig.mandelParams.getMaxIterations()));
            }

            if (tMandelConfig.palette != null) {
                for (int i = 0; i < paletteComboBox.getItemCount(); i++) {
                    if (tMandelConfig.palette.equals((paletteComboBox.getItemAt(i)).getName())) {
                        paletteComboBox.setSelectedIndex(i);
                        break;
                    }
                }
            }

            if (tMandelConfig.paletteConfigJson != null) {
                paletteConfigTextArea.setText(tMandelConfig.paletteConfigJson);
            }
        } finally {
            changeEnabled = true;
        }

        changed();
    }

    public void setChangeListener(final ChangeListener pChangeListener) {
        changeListener = pChangeListener;
    }

    private void changed() {
        if (changeListener != null && changeEnabled) {
            changeListener.stateChanged(null);
        }
    }

    public void setPaletteChangeListener(final ChangeListener pPaletteChangeListener) {
        paletteChangeListener = pPaletteChangeListener;
    }

    private void paletteChanged() {
        if (paletteChangeListener != null && changeEnabled) {
            paletteChangeListener.stateChanged(null);
        }
    }

    public void addPalette(PaletteMapper pPaletteMapper) {
        paletteModel.addElement(pPaletteMapper);
    }

    public PaletteMapper getPaletteMapper() {
        final PaletteMapper tPaletteMapper = (PaletteMapper)paletteComboBox.getSelectedItem();

        final String tText = paletteConfigTextArea.getText();
        if (StringUtils.isNotEmpty(tText)) {
            tPaletteMapper.fromJson(new JSONObject(tText));
        }
        return tPaletteMapper;
    }

    public MandelKernel getSelectedAlgorithm() {
        return (MandelKernel)algorithmComboBox.getSelectedItem();
    }

    public int getMaxIterations() {
        String tMaxIterationsString = (String)maxIterationChooser.getSelectedItem();

        try {
            return Integer.parseInt(tMaxIterationsString);
        } catch (Exception ex) {
            // ignore
        }
        return -1;
    }

    public BigDecimal getX() {
        return getBDOrNull(xTextField.getText());
    }

    public BigDecimal getY() {
        return getBDOrNull(yTextField.getText());
    }

    public BigDecimal getScale() {
        return getBDOrNull(scaleTextField.getText());
    }

    private BigDecimal getBDOrNull(String pText) {
        try {
            return new BigDecimal(pText);
        } catch (Exception ex) {
            return null;
        }
    }

    public void setData(MandelParams pMandelParams) {
        xInfoTextField.setText(Double.toString(pMandelParams.getX_Double()));
        xTextField.setText(null);
        yInfoTextField.setText(Double.toString(pMandelParams.getY_Double()));
        yTextField.setText(null);
        scaleInfoTextField.setText(Double.toString(pMandelParams.getScale_double()));
        scaleTextField.setText(null);
        maxIterInfoTextField.setText(Integer.toString(pMandelParams.getMaxIterations()));
        escapeRadiusTextField.setText(Double.toString(pMandelParams.getEscapeRadius()));
    }

    public void setAlgoInfo(MandelKernel pAlgo) {
        algoInfoTextField.setText(pAlgo != null ? pAlgo.toString() : "");
    }


    public void setRenderMillis(long pCalcMillis, long pColorMillis) {
        renderMillisTextField.setText(String.format("calc: %d color: %d", pCalcMillis, pColorMillis));
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
        mainPanel.setLayout(new FormLayout("fill:max(d;4px):noGrow,left:4dlu:noGrow,fill:d:grow,left:4dlu:noGrow", "center:max(d;4px):noGrow,top:4dlu:noGrow,center:max(d;4px):noGrow,top:4dlu:noGrow,center:max(d;4px):noGrow,top:4dlu:noGrow,center:max(d;4px):noGrow,top:4dlu:noGrow,center:max(d;4px):noGrow,top:4dlu:noGrow,center:max(d;4px):noGrow,top:4dlu:noGrow,center:d:grow,top:4dlu:noGrow,center:max(d;4px):noGrow,top:4dlu:noGrow,center:max(d;4px):noGrow,top:4dlu:noGrow,center:max(d;4px):noGrow,top:4dlu:noGrow,center:max(d;4px):noGrow,top:4dlu:noGrow,center:max(d;4px):noGrow,top:4dlu:noGrow,center:max(d;4px):noGrow,top:4dlu:noGrow,center:max(d;4px):noGrow,top:4dlu:noGrow,center:max(d;4px):noGrow,top:4dlu:noGrow,center:max(d;4px):noGrow,top:4dlu:noGrow,center:max(d;4px):noGrow,top:4dlu:noGrow,center:max(d;4px):noGrow"));
        final JLabel label1 = new JLabel();
        label1.setText("max.Iterations");
        CellConstraints cc = new CellConstraints();
        mainPanel.add(label1, cc.xy(1, 15));
        maxIterationChooser = new JComboBox();
        maxIterationChooser.setEditable(true);
        mainPanel.add(maxIterationChooser, cc.xy(3, 15));
        final JLabel label2 = new JLabel();
        label2.setText("X");
        mainPanel.add(label2, cc.xy(1, 21));
        xTextField = new JTextField();
        xTextField.setColumns(14);
        mainPanel.add(xTextField, cc.xy(3, 21, CellConstraints.FILL, CellConstraints.DEFAULT));
        final JLabel label3 = new JLabel();
        label3.setText("Y");
        mainPanel.add(label3, cc.xy(1, 25));
        yTextField = new JTextField();
        mainPanel.add(yTextField, cc.xy(3, 25, CellConstraints.FILL, CellConstraints.DEFAULT));
        scaleTextField = new JTextField();
        mainPanel.add(scaleTextField, cc.xy(3, 29, CellConstraints.FILL, CellConstraints.DEFAULT));
        final JLabel label4 = new JLabel();
        label4.setText("Scale");
        mainPanel.add(label4, cc.xy(1, 29));
        xInfoTextField = new JTextField();
        xInfoTextField.setEditable(false);
        mainPanel.add(xInfoTextField, cc.xy(3, 23, CellConstraints.FILL, CellConstraints.DEFAULT));
        yInfoTextField = new JTextField();
        yInfoTextField.setEditable(false);
        mainPanel.add(yInfoTextField, cc.xy(3, 27, CellConstraints.FILL, CellConstraints.DEFAULT));
        scaleInfoTextField = new JTextField();
        scaleInfoTextField.setEditable(false);
        mainPanel.add(scaleInfoTextField, cc.xy(3, 31, CellConstraints.FILL, CellConstraints.DEFAULT));
        maxIterInfoTextField = new JTextField();
        maxIterInfoTextField.setEditable(false);
        mainPanel.add(maxIterInfoTextField, cc.xy(3, 17, CellConstraints.FILL, CellConstraints.DEFAULT));
        final JLabel label5 = new JLabel();
        label5.setText("Algo");
        mainPanel.add(label5, cc.xy(1, 5));
        algorithmComboBox = new JComboBox();
        algorithmComboBox.setEditable(false);
        final DefaultComboBoxModel defaultComboBoxModel1 = new DefaultComboBoxModel();
        algorithmComboBox.setModel(defaultComboBoxModel1);
        mainPanel.add(algorithmComboBox, cc.xy(3, 5));
        final JLabel label6 = new JLabel();
        label6.setText("renderTime");
        mainPanel.add(label6, cc.xy(1, 35));
        renderMillisTextField = new JTextField();
        renderMillisTextField.setEditable(false);
        mainPanel.add(renderMillisTextField, cc.xy(3, 35, CellConstraints.FILL, CellConstraints.DEFAULT));
        final JLabel label7 = new JLabel();
        label7.setText("Palette");
        mainPanel.add(label7, cc.xy(1, 11));
        paletteComboBox = new JComboBox();
        paletteComboBox.setEditable(false);
        final DefaultComboBoxModel defaultComboBoxModel2 = new DefaultComboBoxModel();
        paletteComboBox.setModel(defaultComboBoxModel2);
        mainPanel.add(paletteComboBox, cc.xy(3, 11));
        algoInfoTextField = new JTextField();
        algoInfoTextField.setEditable(false);
        mainPanel.add(algoInfoTextField, cc.xy(3, 7, CellConstraints.FILL, CellConstraints.DEFAULT));
        final JLabel label8 = new JLabel();
        label8.setText("Device");
        mainPanel.add(label8, cc.xy(1, 3));
        deviceComboBox = new JComboBox();
        deviceComboBox.setEditable(false);
        final DefaultComboBoxModel defaultComboBoxModel3 = new DefaultComboBoxModel();
        deviceComboBox.setModel(defaultComboBoxModel3);
        mainPanel.add(deviceComboBox, cc.xy(3, 3));
        final JPanel panel1 = new JPanel();
        panel1.setLayout(new FlowLayout(FlowLayout.CENTER, 5, 5));
        mainPanel.add(panel1, cc.xyw(1, 1, 3));
        configsComboBox = new JComboBox();
        panel1.add(configsComboBox);
        addAsConfigButton = new JButton();
        addAsConfigButton.setText("add");
        panel1.add(addAsConfigButton);
        removeSelectedConfig = new JButton();
        removeSelectedConfig.setText("del");
        panel1.add(removeSelectedConfig);
        final JLabel label9 = new JLabel();
        label9.setText("Escape Radius");
        mainPanel.add(label9, cc.xy(1, 19));
        escapeRadiusTextField = new JTextField();
        escapeRadiusTextField.setColumns(14);
        mainPanel.add(escapeRadiusTextField, cc.xy(3, 19, CellConstraints.FILL, CellConstraints.DEFAULT));
        final JScrollPane scrollPane1 = new JScrollPane();
        mainPanel.add(scrollPane1, cc.xy(3, 13, CellConstraints.FILL, CellConstraints.FILL));
        paletteConfigTextArea = new JTextArea();
        paletteConfigTextArea.setRows(3);
        paletteConfigTextArea.setText("");
        scrollPane1.setViewportView(paletteConfigTextArea);
        final JLabel label10 = new JLabel();
        label10.setText("Zoom-Speed");
        mainPanel.add(label10, cc.xy(1, 33));
        zoomSpeedSlider = new JSlider();
        zoomSpeedSlider.setMaximum(10);
        zoomSpeedSlider.setMinimum(1);
        zoomSpeedSlider.setValue(5);
        mainPanel.add(zoomSpeedSlider, cc.xy(3, 33, CellConstraints.FILL, CellConstraints.DEFAULT));
        calcDistanceCheckBox = new JCheckBox();
        calcDistanceCheckBox.setText("");
        mainPanel.add(calcDistanceCheckBox, cc.xy(3, 9));
        final JLabel label11 = new JLabel();
        label11.setText("Calc Distance");
        label11.setToolTipText("About 20% faster without -> no Distance palette mapping");
        mainPanel.add(label11, cc.xy(1, 9));
    }

    /**
     * @noinspection ALL
     */
    public JComponent $$$getRootComponent$$$() {
        return mainPanel;
    }

    public JComponent getComponent() {
        return mainPanel;
    }

    public void setAlgorithms(final MandelKernel... pAlgos) {
        MandelKernel tSelection = getSelectedAlgorithm();
        algorithmModel.removeAllElements();
        algorithmModel.addElement(null);
        for (MandelKernel tAlgo : pAlgos) {
            algorithmModel.addElement(tAlgo);

            if (tSelection != null && tSelection.toString().equals(tAlgo.toString())) {
                tSelection = tAlgo;
            }
        }

        if (tSelection != null) {
            algorithmComboBox.setSelectedItem(tSelection);
        } else {
            algorithmComboBox.setSelectedItem(null);
        }
    }

    public Device getDevice() {
        Device tSelectedDevice = (Device) deviceComboBox.getSelectedItem();
        if ( tSelectedDevice==null ) {
            return JavaDevice.THREAD_POOL;
        }
        return (Device) tSelectedDevice;
    }


    private void saveConfigs(Path pDevicesFile) {
        try {
            JSONObject tFileData = new JSONObject();
            final JSONObject tConfigs = new JSONObject();
            tFileData.put("mandelConfigs", tConfigs);

            for (MandelConfig tConfig : configs.values()) {
                tConfigs.put(tConfig.name, tConfig.toJson());
            }

            Files.write(pDevicesFile, tFileData.toString(2).getBytes(), StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING);
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(addAsConfigButton, "cannot store configuration " + ex.getMessage());
            ex.printStackTrace();
        }

    }

    private void loadConfig(Path pDevicesFile) throws IOException {
        configs.clear();

        final JSONObject tFileData = new JSONObject(new String(Files.readAllBytes(pDevicesFile)));

        final JSONObject tConfigsJson = tFileData.getJSONObject("mandelConfigs");

        configsComboBox.removeAllItems();
        final List<String> tConfigNames = new ArrayList<>(tConfigsJson.keySet());
        tConfigNames.sort(String::compareToIgnoreCase);
        for (String tKey : tConfigNames) {
            MandelConfig tMandelConfig = new MandelConfig();
            tMandelConfig.fromJson(tConfigsJson.getJSONObject(tKey));
            if (configs.put(tMandelConfig.name, tMandelConfig) == null) {
                configsComboBox.addItem(tMandelConfig);
            }
        }
        configsComboBox.setSelectedItem(null);
    }

    public int getZoomSpeed() {
        return zoomSpeedSlider.getValue();
    }

    static class MandelConfig {
        private String name;

        private MandelParams mandelParams;

        private String palette;
        private String paletteConfigJson;

        public JSONObject toJson() {
            JSONObject tJSONObject = new JSONObject();
            tJSONObject.put("coords", mandelParams.toJson());
            tJSONObject.put("name", name);
            tJSONObject.put("palette", palette);
            if (StringUtils.isNotEmpty(paletteConfigJson)) {
                tJSONObject.put("paletteConfigJson", new JSONObject(paletteConfigJson));
            }

            return tJSONObject;
        }

        public void fromJson(JSONObject pJSONObject) {
            mandelParams = new MandelParams();
            mandelParams.fromJson(pJSONObject.getJSONObject("coords"));
            name = pJSONObject.getString("name");
            if (pJSONObject.has("palette")) {
                palette = pJSONObject.getString("palette");
            }
            if (pJSONObject.has("paletteConfigJson")) {
                JSONObject tPaletteConfig = pJSONObject.getJSONObject("paletteConfigJson");
                paletteConfigJson = tPaletteConfig.toString(2);
            }
        }

        @Override
        public String toString() {
            return name;
        }
    }

}
