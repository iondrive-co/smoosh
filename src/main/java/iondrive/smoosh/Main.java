package iondrive.smoosh;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;

import java.util.List;

public class Main {
    public static void main(final String[] args) {
        if (args.length < 1) {
            System.out.println("Smoosh - Modern Person Detection");
            System.out.println("\nUsage: java -jar smoosh.jar <image-path>");
            System.out.println("\nExample:");
            System.out.println("  java -jar smoosh.jar photo.jpg");
            System.out.println("\nThis will display the image with bounding boxes around detected people.");
            System.out.println("\nFor library usage:");
            System.out.println("  PersonDetector detector = new PersonDetector();");
            System.out.println("  List<Detection> people = detector.detectPeople(\"image.jpg\");");
            return;
        }

        final String imagePath = args[0];

        try (final PersonDetector detector = new PersonDetector()) {
            System.out.println("Detecting people in: " + imagePath);

            final List<PersonDetector.Detection> detections = detector.detectPeople(imagePath);

            // Load image for visualization
            final Mat image = Imgcodecs.imread(imagePath);
            if (image.empty()) {
                System.err.println("Error: Could not load image: " + imagePath);
                System.exit(1);
            }

            // Draw bounding boxes
            for (int i = 0; i < detections.size(); i++) {
                final PersonDetector.Detection detection = detections.get(i);

                // Draw green rectangle
                final Point topLeft = new Point(detection.bbox.x, detection.bbox.y);
                final Point bottomRight = new Point(
                    detection.bbox.x + detection.bbox.width,
                    detection.bbox.y + detection.bbox.height
                );
                Imgproc.rectangle(image, topLeft, bottomRight, new Scalar(0, 255, 0), 2);

                // Draw label with confidence
                final String label = String.format("Person %d: %.1f%%", i + 1, detection.confidence * 100);
                final Point labelPos = new Point(detection.bbox.x, detection.bbox.y - 10);
                Imgproc.putText(image, label, labelPos, Imgproc.FONT_HERSHEY_SIMPLEX,
                              0.6, new Scalar(0, 255, 0), 2);
            }

            // Display result
            final String title = detections.isEmpty()
                ? "No people detected"
                : String.format("Found %d person(s) - Press any key to close", detections.size());

            System.out.println(detections.isEmpty()
                ? "No people detected."
                : String.format("Found %d person(s) - displaying image...", detections.size()));

            HighGui.imshow(title, image);
            HighGui.waitKey(0);
            HighGui.destroyAllWindows();

        } catch (final Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
