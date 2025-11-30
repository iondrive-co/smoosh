package iondrive.smoosh;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;

public class RegionDetector {

    static {
        nu.pattern.OpenCV.loadLocally();
    }

    public enum DetectionMethod {
        EDGE,
        SALIENCY,
        FACE
    }

    public Rectangle detectRegionOfInterest(final String imagePath, final DetectionMethod method) {
        final Mat image = Imgcodecs.imread(imagePath);
        if (image.empty()) {
            throw new RuntimeException("Failed to load image: " + imagePath);
        }

        return switch (method) {
            case EDGE -> detectUsingEdges(image);
            case SALIENCY -> detectUsingSaliency(image);
            case FACE -> detectUsingFace(image);
        };
    }

    private Rectangle detectUsingEdges(final Mat image) {
        final Mat gray = new Mat();
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);

        final Mat blurred = new Mat();
        Imgproc.GaussianBlur(gray, blurred, new Size(5, 5), 0);

        final Mat edges = new Mat();
        Imgproc.Canny(blurred, edges, 50, 150);

        final List<MatOfPoint> contours = new ArrayList<>();
        final Mat hierarchy = new Mat();
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        if (contours.isEmpty()) {
            return new Rectangle(0, 0, image.width(), image.height());
        }

        MatOfPoint largestContour = contours.get(0);
        double maxArea = Imgproc.contourArea(largestContour);

        for (final MatOfPoint contour : contours) {
            final double area = Imgproc.contourArea(contour);
            if (area > maxArea) {
                maxArea = area;
                largestContour = contour;
            }
        }

        final Rect boundingRect = Imgproc.boundingRect(largestContour);

        return new Rectangle(boundingRect.x, boundingRect.y, boundingRect.width, boundingRect.height);
    }

    private Rectangle detectUsingSaliency(final Mat image) {
        final Mat resized;
        final Size targetSize = new Size(640, 480);
        final double scale = Math.min(targetSize.width / image.width(), targetSize.height / image.height());

        if (scale < 1.0) {
            resized = new Mat();
            Imgproc.resize(image, resized, new Size(image.width() * scale, image.height() * scale));
        } else {
            resized = image;
        }

        final Mat gray = new Mat();
        Imgproc.cvtColor(resized, gray, Imgproc.COLOR_BGR2GRAY);

        final Mat blurred = new Mat();
        Imgproc.GaussianBlur(gray, blurred, new Size(9, 9), 0);

        final Mat floatGray = new Mat();
        blurred.convertTo(floatGray, CvType.CV_32F);

        final Mat laplacian = new Mat();
        Imgproc.Laplacian(floatGray, laplacian, CvType.CV_32F);
        Core.convertScaleAbs(laplacian, laplacian);

        final Mat binary = new Mat();
        Imgproc.threshold(laplacian, binary, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
        binary.convertTo(binary, CvType.CV_8U);

        final List<MatOfPoint> contours = new ArrayList<>();
        final Mat hierarchy = new Mat();
        Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        if (contours.isEmpty()) {
            return new Rectangle(0, 0, image.width(), image.height());
        }

        MatOfPoint largestContour = contours.get(0);
        double maxArea = Imgproc.contourArea(largestContour);

        for (final MatOfPoint contour : contours) {
            final double area = Imgproc.contourArea(contour);
            if (area > maxArea) {
                maxArea = area;
                largestContour = contour;
            }
        }

        final Rect boundingRect = Imgproc.boundingRect(largestContour);

        if (scale < 1.0) {
            final int x = (int) (boundingRect.x / scale);
            final int y = (int) (boundingRect.y / scale);
            final int width = (int) (boundingRect.width / scale);
            final int height = (int) (boundingRect.height / scale);
            return new Rectangle(x, y, width, height);
        }

        return new Rectangle(boundingRect.x, boundingRect.y, boundingRect.width, boundingRect.height);
    }

    private Rectangle detectUsingFace(final Mat image) {
        final Mat gray = new Mat();
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);

        Imgproc.equalizeHist(gray, gray);

        final CascadeClassifier faceDetector = new CascadeClassifier();

        final String cascadePath = getClass().getClassLoader().getResource("haarcascade_frontalface_default.xml").getPath();

        if (!faceDetector.load(cascadePath)) {
            throw new RuntimeException("Failed to load Haar Cascade classifier");
        }

        final MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(gray, faceDetections);

        final Rect[] faces = faceDetections.toArray();

        if (faces.length == 0) {
            return new Rectangle(0, 0, image.width(), image.height());
        }

        Rect largestFace = faces[0];
        int maxArea = largestFace.width * largestFace.height;

        for (final Rect face : faces) {
            final int area = face.width * face.height;
            if (area > maxArea) {
                maxArea = area;
                largestFace = face;
            }
        }

        return new Rectangle(largestFace.x, largestFace.y, largestFace.width, largestFace.height);
    }

    public static void main(final String[] args) {
        if (args.length < 1) {
            System.out.println("Usage: java -jar smoosh.jar <image-path> [method]");
            System.out.println("Methods: EDGE, SALIENCY, FACE (default: EDGE)");
            System.out.println("\nExample:");
            System.out.println("  java -jar smoosh.jar image.jpg EDGE");
            System.out.println("  java -jar smoosh.jar image.jpg SALIENCY");
            System.out.println("  java -jar smoosh.jar image.jpg FACE");
            return;
        }

        final String imagePath = args[0];
        DetectionMethod method = DetectionMethod.EDGE;

        if (args.length >= 2) {
            try {
                method = DetectionMethod.valueOf(args[1].toUpperCase());
            } catch (final IllegalArgumentException e) {
                System.err.println("Invalid method: " + args[1]);
                System.err.println("Valid methods: EDGE, SALIENCY, FACE");
                return;
            }
        }

        try {
            final RegionDetector detector = new RegionDetector();
            System.out.println("Detecting region of interest using method: " + method);

            final Rectangle region = detector.detectRegionOfInterest(imagePath, method);

            System.out.println("\nResult:");
            System.out.println("Region of Interest: " + formatRectangle(region));
            System.out.println("\nCoordinates:");
            System.out.println("  Top-left corner: (" + region.x + ", " + region.y + ")");
            System.out.println("  Dimensions: " + region.width + "x" + region.height);

        } catch (final Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static String formatRectangle(final Rectangle r) {
        return String.format("Rectangle[x=%d, y=%d, width=%d, height=%d]", r.x, r.y, r.width, r.height);
    }
}
