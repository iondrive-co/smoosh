package iondrive.smoosh;

import javax.swing.*;
import java.awt.*;
import java.io.File;

public class RegionDetectorGui {

    public static void main(final String[] args) {
        SwingUtilities.invokeLater(() -> {
            try {
                UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
            } catch (Exception e) {
                // Use default look and feel if system one fails
            }

            final JFileChooser fileChooser = new JFileChooser();
            fileChooser.setDialogTitle("Select Image File");
            fileChooser.setFileFilter(new javax.swing.filechooser.FileNameExtensionFilter(
                "Image files", "jpg", "jpeg", "png", "bmp", "gif"));

            final int result = fileChooser.showOpenDialog(null);

            if (result != JFileChooser.APPROVE_OPTION) {
                System.out.println("No file selected. Exiting.");
                return;
            }

            final File selectedFile = fileChooser.getSelectedFile();
            final String imagePath = selectedFile.getAbsolutePath();

            final RegionDetector.DetectionMethod[] methods = RegionDetector.DetectionMethod.values();
            final String[] methodNames = new String[methods.length];
            for (int i = 0; i < methods.length; i++) {
                methodNames[i] = methods[i].toString();
            }

            final String selectedMethod = (String) JOptionPane.showInputDialog(
                null,
                "Select detection method:",
                "Detection Method",
                JOptionPane.QUESTION_MESSAGE,
                null,
                methodNames,
                methodNames[0]);

            if (selectedMethod == null) {
                System.out.println("No method selected. Exiting.");
                return;
            }

            final RegionDetector.DetectionMethod method = RegionDetector.DetectionMethod.valueOf(selectedMethod);

            try {
                final RegionDetector detector = new RegionDetector();
                System.out.println("Processing image: " + imagePath);
                System.out.println("Using detection method: " + method);

                final Rectangle region = detector.detectRegionOfInterest(imagePath, method);

                final String message = String.format(
                    "Region of Interest detected:\n\n" +
                    "Position: (%d, %d)\n" +
                    "Size: %d x %d pixels\n\n" +
                    "Image: %s",
                    region.x, region.y, region.width, region.height,
                    selectedFile.getName());

                System.out.println("\n" + message.replace("\n\n", "\n"));

                JOptionPane.showMessageDialog(
                    null,
                    message,
                    "Detection Result",
                    JOptionPane.INFORMATION_MESSAGE);

            } catch (final Exception e) {
                final String errorMessage = "Error processing image: " + e.getMessage();
                System.err.println(errorMessage);
                e.printStackTrace();

                JOptionPane.showMessageDialog(
                    null,
                    errorMessage,
                    "Error",
                    JOptionPane.ERROR_MESSAGE);
            }
        });
    }
}
