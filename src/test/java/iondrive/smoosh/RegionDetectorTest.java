package iondrive.smoosh;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.awt.*;
import java.nio.file.Path;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class RegionDetectorTest {

    private static final int TOLERANCE = 20;

    @BeforeAll
    static void setup() {
        nu.pattern.OpenCV.loadLocally();
    }

    @Test
    void edgeDetection_shouldFindBlackRectangleOnWhiteBackground(@TempDir final Path tempDir) {
        final int imgWidth = 600, imgHeight = 400;
        final int rectX = 150, rectY = 100, rectWidth = 300, rectHeight = 200;

        final String imagePath = createImageWithRectangle(tempDir, "black_rect.jpg",
            imgWidth, imgHeight, rectX, rectY, rectWidth, rectHeight,
            new Scalar(255, 255, 255), new Scalar(0, 0, 0));

        final RegionDetector detector = new RegionDetector();
        final Rectangle region = detector.detectRegionOfInterest(imagePath, RegionDetector.DetectionMethod.EDGE);

        assertRegionApproximatelyEquals(region, rectX, rectY, rectWidth, rectHeight, TOLERANCE);
    }

    @Test
    void edgeDetection_shouldFindLargestOfMultipleRectangles(@TempDir final Path tempDir) {
        final String imagePath = createImageWithMultipleRectangles(tempDir, "multi_rect.jpg");

        final RegionDetector detector = new RegionDetector();
        final Rectangle region = detector.detectRegionOfInterest(imagePath, RegionDetector.DetectionMethod.EDGE);

        final int largeRectX = 100, largeRectY = 50, largeRectWidth = 400, largeRectHeight = 300;
        assertRegionApproximatelyEquals(region, largeRectX, largeRectY, largeRectWidth, largeRectHeight, TOLERANCE);
    }

    @Test
    void edgeDetection_shouldDetectCircle(@TempDir final Path tempDir) {
        final int imgWidth = 500, imgHeight = 500;
        final int centerX = 250, centerY = 250, radius = 100;

        final String imagePath = createImageWithCircle(tempDir, "circle.jpg",
            imgWidth, imgHeight, centerX, centerY, radius);

        final RegionDetector detector = new RegionDetector();
        final Rectangle region = detector.detectRegionOfInterest(imagePath, RegionDetector.DetectionMethod.EDGE);

        final int expectedX = centerX - radius;
        final int expectedY = centerY - radius;
        final int expectedSize = radius * 2;

        assertRegionApproximatelyEquals(region, expectedX, expectedY, expectedSize, expectedSize, TOLERANCE);
    }

    @Test
    void edgeDetection_shouldReturnFullImageForUniformBackground(@TempDir final Path tempDir) {
        final int imgWidth = 640, imgHeight = 480;
        final String imagePath = createUniformImage(tempDir, "uniform.jpg", imgWidth, imgHeight, new Scalar(128, 128, 128));

        final RegionDetector detector = new RegionDetector();
        final Rectangle region = detector.detectRegionOfInterest(imagePath, RegionDetector.DetectionMethod.EDGE);

        assertEquals(imgWidth, region.width);
        assertEquals(imgHeight, region.height);
        assertEquals(0, region.x);
        assertEquals(0, region.y);
    }

    @Test
    void saliencyDetection_shouldFindBrightRedObjectOnDarkBackground(@TempDir final Path tempDir) {
        final int imgWidth = 800, imgHeight = 600;
        final int objX = 200, objY = 150, objWidth = 400, objHeight = 300;

        final String imagePath = createImageWithRectangle(tempDir, "bright_object.jpg",
            imgWidth, imgHeight, objX, objY, objWidth, objHeight,
            new Scalar(20, 20, 20), new Scalar(0, 0, 255));

        final RegionDetector detector = new RegionDetector();
        final Rectangle region = detector.detectRegionOfInterest(imagePath, RegionDetector.DetectionMethod.SALIENCY);

        assertTrue(region.x >= objX - TOLERANCE * 2);
        assertTrue(region.y >= objY - TOLERANCE * 2);
        assertTrue(region.x + region.width <= objX + objWidth + TOLERANCE * 2);
        assertTrue(region.y + region.height <= objY + objHeight + TOLERANCE * 2);
    }

    @Test
    void saliencyDetection_shouldFindHighContrastRegion(@TempDir final Path tempDir) {
        final String imagePath = createCheckerboardInCenter(tempDir, "checkerboard.jpg");

        final RegionDetector detector = new RegionDetector();
        final Rectangle region = detector.detectRegionOfInterest(imagePath, RegionDetector.DetectionMethod.SALIENCY);

        assertTrue(region.width > 0 && region.width < 800);
        assertTrue(region.height > 0 && region.height < 600);
        assertTrue(region.x > 50);
        assertTrue(region.y > 50);
    }

    @Test
    void edgeDetection_shouldHandleNoisyImage(@TempDir final Path tempDir) {
        final int imgWidth = 640, imgHeight = 480;
        final int rectX = 200, rectY = 150, rectWidth = 240, rectHeight = 180;

        final String imagePath = createNoisyImageWithRectangle(tempDir, "noisy.jpg",
            imgWidth, imgHeight, rectX, rectY, rectWidth, rectHeight);

        final RegionDetector detector = new RegionDetector();
        final Rectangle region = detector.detectRegionOfInterest(imagePath, RegionDetector.DetectionMethod.EDGE);

        assertNotNull(region);
        assertTrue(region.width > 0);
        assertTrue(region.height > 0);
    }

    @Test
    void edgeDetection_shouldDetectWhiteRectangleOnBlackBackground(@TempDir final Path tempDir) {
        final int imgWidth = 500, imgHeight = 400;
        final int rectX = 100, rectY = 80, rectWidth = 300, rectHeight = 240;

        final String imagePath = createImageWithRectangle(tempDir, "white_rect.jpg",
            imgWidth, imgHeight, rectX, rectY, rectWidth, rectHeight,
            new Scalar(0, 0, 0), new Scalar(255, 255, 255));

        final RegionDetector detector = new RegionDetector();
        final Rectangle region = detector.detectRegionOfInterest(imagePath, RegionDetector.DetectionMethod.EDGE);

        assertRegionApproximatelyEquals(region, rectX, rectY, rectWidth, rectHeight, TOLERANCE);
    }

    @Test
    void edgeDetection_shouldDetectOffCenterRectangle(@TempDir final Path tempDir) {
        final int imgWidth = 1000, imgHeight = 800;
        final int rectX = 700, rectY = 600, rectWidth = 200, rectHeight = 150;

        final String imagePath = createImageWithRectangle(tempDir, "offset_rect.jpg",
            imgWidth, imgHeight, rectX, rectY, rectWidth, rectHeight,
            new Scalar(255, 255, 255), new Scalar(0, 0, 0));

        final RegionDetector detector = new RegionDetector();
        final Rectangle region = detector.detectRegionOfInterest(imagePath, RegionDetector.DetectionMethod.EDGE);

        assertTrue(region.x >= rectX - TOLERANCE);
        assertTrue(region.y >= rectY - TOLERANCE);
    }

    @Test
    void saliencyDetection_shouldPreferHigherContrastRegion(@TempDir final Path tempDir) {
        final String imagePath = createImageWithTwoRegionsDifferentContrast(tempDir, "contrast.jpg");

        final RegionDetector detector = new RegionDetector();
        final Rectangle region = detector.detectRegionOfInterest(imagePath, RegionDetector.DetectionMethod.SALIENCY);

        assertTrue(region.x >= 400 - TOLERANCE * 3);
    }

    @Test
    void edgeDetection_shouldDetectColoredShapeOnColoredBackground(@TempDir final Path tempDir) {
        final int imgWidth = 600, imgHeight = 400;
        final int rectX = 150, rectY = 100, rectWidth = 300, rectHeight = 200;

        final String imagePath = createImageWithRectangle(tempDir, "colored.jpg",
            imgWidth, imgHeight, rectX, rectY, rectWidth, rectHeight,
            new Scalar(100, 150, 200), new Scalar(200, 50, 50));

        final RegionDetector detector = new RegionDetector();
        final Rectangle region = detector.detectRegionOfInterest(imagePath, RegionDetector.DetectionMethod.EDGE);

        assertRegionApproximatelyEquals(region, rectX, rectY, rectWidth, rectHeight, TOLERANCE * 2);
    }

    @Test
    void nonexistentFile_shouldThrowException() {
        final RegionDetector detector = new RegionDetector();

        final RuntimeException exception = assertThrows(RuntimeException.class, () -> {
            detector.detectRegionOfInterest("/nonexistent/image.jpg", RegionDetector.DetectionMethod.EDGE);
        });

        assertTrue(exception.getMessage().contains("Failed to load image"));
    }

    private String createImageWithRectangle(final Path tempDir, final String filename,
                                           final int imgWidth, final int imgHeight,
                                           final int rectX, final int rectY, final int rectWidth, final int rectHeight,
                                           final Scalar bgColor, final Scalar rectColor) {
        final String path = tempDir.resolve(filename).toString();
        final Mat image = new Mat(imgHeight, imgWidth, CvType.CV_8UC3, bgColor);

        Imgproc.rectangle(image,
            new Point(rectX, rectY),
            new Point(rectX + rectWidth, rectY + rectHeight),
            rectColor, -1);

        Imgcodecs.imwrite(path, image);
        return path;
    }

    private String createImageWithMultipleRectangles(final Path tempDir, final String filename) {
        final String path = tempDir.resolve(filename).toString();
        final Mat image = new Mat(400, 600, CvType.CV_8UC3, new Scalar(255, 255, 255));

        Imgproc.rectangle(image, new Point(100, 50), new Point(500, 350),
            new Scalar(0, 0, 0), -1);

        Imgproc.rectangle(image, new Point(10, 10), new Point(60, 60),
            new Scalar(50, 50, 50), -1);

        Imgproc.rectangle(image, new Point(530, 300), new Point(590, 380),
            new Scalar(100, 100, 100), -1);

        Imgcodecs.imwrite(path, image);
        return path;
    }

    private String createImageWithCircle(final Path tempDir, final String filename,
                                        final int imgWidth, final int imgHeight,
                                        final int centerX, final int centerY, final int radius) {
        final String path = tempDir.resolve(filename).toString();
        final Mat image = new Mat(imgHeight, imgWidth, CvType.CV_8UC3, new Scalar(255, 255, 255));

        Imgproc.circle(image, new Point(centerX, centerY), radius, new Scalar(0, 0, 0), -1);

        Imgcodecs.imwrite(path, image);
        return path;
    }

    private String createUniformImage(final Path tempDir, final String filename,
                                     final int width, final int height, final Scalar color) {
        final String path = tempDir.resolve(filename).toString();
        final Mat image = new Mat(height, width, CvType.CV_8UC3, color);
        Imgcodecs.imwrite(path, image);
        return path;
    }

    private String createCheckerboardInCenter(final Path tempDir, final String filename) {
        final String path = tempDir.resolve(filename).toString();
        final Mat image = new Mat(600, 800, CvType.CV_8UC3, new Scalar(128, 128, 128));

        final int startX = 250, startY = 200, size = 300;
        final int squareSize = 30;

        for (int y = 0; y < size; y += squareSize) {
            for (int x = 0; x < size; x += squareSize) {
                final boolean isBlack = ((x / squareSize) + (y / squareSize)) % 2 == 0;
                final Scalar color = isBlack ? new Scalar(0, 0, 0) : new Scalar(255, 255, 255);

                Imgproc.rectangle(image,
                    new Point(startX + x, startY + y),
                    new Point(startX + x + squareSize, startY + y + squareSize),
                    color, -1);
            }
        }

        Imgcodecs.imwrite(path, image);
        return path;
    }

    private String createNoisyImageWithRectangle(final Path tempDir, final String filename,
                                                 final int imgWidth, final int imgHeight,
                                                 final int rectX, final int rectY, final int rectWidth, final int rectHeight) {
        final String path = tempDir.resolve(filename).toString();
        final Mat image = new Mat(imgHeight, imgWidth, CvType.CV_8UC3);

        final Random rand = new Random(42);
        for (int y = 0; y < imgHeight; y++) {
            for (int x = 0; x < imgWidth; x++) {
                final int value = rand.nextInt(50) + 200;
                image.put(y, x, value, value, value);
            }
        }

        Imgproc.rectangle(image,
            new Point(rectX, rectY),
            new Point(rectX + rectWidth, rectY + rectHeight),
            new Scalar(0, 0, 0), -1);

        Imgcodecs.imwrite(path, image);
        return path;
    }

    private String createImageWithTwoRegionsDifferentContrast(final Path tempDir, final String filename) {
        final String path = tempDir.resolve(filename).toString();
        final Mat image = new Mat(400, 800, CvType.CV_8UC3, new Scalar(128, 128, 128));

        Imgproc.rectangle(image,
            new Point(50, 100),
            new Point(250, 300),
            new Scalar(100, 100, 100), -1);

        Imgproc.rectangle(image,
            new Point(450, 100),
            new Point(750, 300),
            new Scalar(255, 255, 255), -1);

        Imgcodecs.imwrite(path, image);
        return path;
    }

    private void assertRegionApproximatelyEquals(final Rectangle actual,
                                                 final int expectedX, final int expectedY,
                                                 final int expectedWidth, final int expectedHeight,
                                                 final int tolerance) {
        assertTrue(Math.abs(actual.x - expectedX) <= tolerance,
            String.format("Expected x=%d, got x=%d (tolerance=%d)", expectedX, actual.x, tolerance));
        assertTrue(Math.abs(actual.y - expectedY) <= tolerance,
            String.format("Expected y=%d, got y=%d (tolerance=%d)", expectedY, actual.y, tolerance));
        assertTrue(Math.abs(actual.width - expectedWidth) <= tolerance,
            String.format("Expected width=%d, got width=%d (tolerance=%d)", expectedWidth, actual.width, tolerance));
        assertTrue(Math.abs(actual.height - expectedHeight) <= tolerance,
            String.format("Expected height=%d, got height=%d (tolerance=%d)", expectedHeight, actual.height, tolerance));
    }
}
