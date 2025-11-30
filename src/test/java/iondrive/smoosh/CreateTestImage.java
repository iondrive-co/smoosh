package iondrive.smoosh;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class CreateTestImage {
    static {
        nu.pattern.OpenCV.loadLocally();
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
        System.out.println("Created test image: " + filename);
    }
}
