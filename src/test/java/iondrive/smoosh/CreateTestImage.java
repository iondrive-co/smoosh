package iondrive.smoosh;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.nio.file.Path;

/**
 * Utility for creating test images.
 */
public class CreateTestImage {
    static {
        nu.pattern.OpenCV.loadLocally();
    }

    /**
     * Create a uniform (solid color) test image.
     * Useful for testing edge cases where no person is present.
     */
    public static String createUniformImage(final Path tempDir, final String filename,
                                           final int width, final int height) {
        final String path = tempDir.resolve(filename).toString();
        final Mat image = new Mat(height, width, CvType.CV_8UC3, new Scalar(128, 128, 128));
        Imgcodecs.imwrite(path, image);
        image.release();
        return path;
    }

    public static void main(final String[] args) {
        final String filename = args.length > 0 ? args[0] : "test image with spaces.jpg";

        final Mat image = new Mat(400, 600, CvType.CV_8UC3, new Scalar(255, 255, 255));

        Imgproc.rectangle(image,
            new Point(150, 100),
            new Point(450, 300),
            new Scalar(0, 0, 255),
            -1);

        Imgcodecs.imwrite(filename, image);
        image.release();
        System.out.println("Created test image: " + filename);
    }
}
