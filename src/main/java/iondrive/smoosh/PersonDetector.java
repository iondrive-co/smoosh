package iondrive.smoosh;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.awt.Rectangle;
import java.io.File;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.*;

/**
 * Modern person detection using YOLOv8 via ONNX Runtime.
 * Simple API: just detect people in images.
 */
public class PersonDetector implements AutoCloseable {

    static {
        nu.pattern.OpenCV.loadLocally();
    }

    private final OrtEnvironment env;
    private final OrtSession session;
    private final int inputSize = 640;
    private final float confThreshold = 0.25f;
    private final float iouThreshold = 0.45f;

    public static class Detection {
        public final Rectangle bbox;
        public final float confidence;

        public Detection(final Rectangle bbox, final float confidence) {
            this.bbox = bbox;
            this.confidence = confidence;
        }

        @Override
        public String toString() {
            return String.format("Person[bbox=%s, conf=%.2f]", bbox, confidence);
        }
    }

    public PersonDetector() {
        try {
            env = OrtEnvironment.getEnvironment();
            final String modelPath = extractModel();
            session = env.createSession(modelPath, new OrtSession.SessionOptions());
        } catch (final Exception e) {
            throw new RuntimeException("Failed to initialize PersonDetector: " + e.getMessage(), e);
        }
    }

    /**
     * Detect all people in an image.
     * @param imagePath Path to the image file
     * @return List of detected people with bounding boxes and confidence scores
     */
    public List<Detection> detectPeople(final String imagePath) {
        final Mat image = Imgcodecs.imread(imagePath);
        if (image.empty()) {
            throw new RuntimeException("Failed to load image: " + imagePath);
        }

        try {
            return detectPeople(image);
        } finally {
            image.release();
        }
    }

    /**
     * Detect all people in an OpenCV Mat image.
     */
    public List<Detection> detectPeople(final Mat image) {
        try {
            final Mat preprocessed = preprocessImage(image);
            final float[] inputData = matToFloatArray(preprocessed);
            preprocessed.release();

            final long[] inputShape = {1, 3, inputSize, inputSize};
            final OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), inputShape);

            final Map<String, OnnxTensor> inputs = Map.of("images", inputTensor);
            final OrtSession.Result results = session.run(inputs);

            // YOLOv8 output shape is (1, 84, 8400) - batch, features, detections
            final float[][][] rawOutput = (float[][][]) results.get(0).getValue();
            final float[][] output = rawOutput[0]; // Remove batch dimension
            inputTensor.close();

            return postprocess(output, image.width(), image.height());

        } catch (final Exception e) {
            throw new RuntimeException("Detection failed: " + e.getMessage(), e);
        }
    }

    /**
     * Detect largest person in image (convenience method).
     * @return Rectangle of largest person, or full image rect if none found
     */
    public Rectangle detectLargestPerson(final String imagePath) {
        final List<Detection> detections = detectPeople(imagePath);

        if (detections.isEmpty()) {
            final Mat image = Imgcodecs.imread(imagePath);
            final Rectangle fallback = new Rectangle(0, 0, image.width(), image.height());
            image.release();
            return fallback;
        }

        return detections.stream()
            .max(Comparator.comparingInt(d -> d.bbox.width * d.bbox.height))
            .map(d -> d.bbox)
            .orElseThrow();
    }

    private Mat preprocessImage(final Mat image) {
        final Mat resized = new Mat();
        Imgproc.resize(image, resized, new Size(inputSize, inputSize));

        final Mat rgb = new Mat();
        Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_BGR2RGB);
        resized.release();

        rgb.convertTo(rgb, CvType.CV_32FC3, 1.0 / 255.0);

        return rgb;
    }

    private float[] matToFloatArray(final Mat mat) {
        final int channels = mat.channels();
        final int height = mat.height();
        final int width = mat.width();
        final float[] data = new float[channels * height * width];

        final float[] pixel = new float[channels];
        int idx = 0;

        for (int c = 0; c < channels; c++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    mat.get(y, x, pixel);
                    data[idx++] = pixel[c];
                }
            }
        }

        return data;
    }

    private List<Detection> postprocess(final float[][] output, final int origWidth, final int origHeight) {
        final List<Detection> detections = new ArrayList<>();

        // YOLOv8 output format: (84, 8400)
        // Rows 0-3: x, y, w, h (bounding box)
        // Rows 4-83: class probabilities for 80 COCO classes
        final int numDetections = output[0].length;

        for (int i = 0; i < numDetections; i++) {
            final float x = output[0][i];
            final float y = output[1][i];
            final float w = output[2][i];
            final float h = output[3][i];

            float maxConf = 0;
            int maxClass = -1;

            // Find class with highest confidence
            for (int c = 4; c < output.length; c++) {
                final float conf = output[c][i];
                if (conf > maxConf) {
                    maxConf = conf;
                    maxClass = c - 4;
                }
            }

            // Class 0 is "person" in COCO dataset
            if (maxClass == 0 && maxConf >= confThreshold) {
                final float xScale = (float) origWidth / inputSize;
                final float yScale = (float) origHeight / inputSize;

                final int x1 = (int) ((x - w / 2) * xScale);
                final int y1 = (int) ((y - h / 2) * yScale);
                final int x2 = (int) ((x + w / 2) * xScale);
                final int y2 = (int) ((y + h / 2) * yScale);

                final Rectangle bbox = new Rectangle(
                    Math.max(0, x1),
                    Math.max(0, y1),
                    Math.min(x2 - x1, origWidth),
                    Math.min(y2 - y1, origHeight)
                );

                detections.add(new Detection(bbox, maxConf));
            }
        }

        return applyNMS(detections);
    }

    private List<Detection> applyNMS(final List<Detection> detections) {
        detections.sort((a, b) -> Float.compare(b.confidence, a.confidence));

        final List<Detection> result = new ArrayList<>();

        for (final Detection detection : detections) {
            boolean keep = true;

            for (final Detection kept : result) {
                if (calculateIoU(detection.bbox, kept.bbox) > iouThreshold) {
                    keep = false;
                    break;
                }
            }

            if (keep) {
                result.add(detection);
            }
        }

        return result;
    }

    private float calculateIoU(final Rectangle a, final Rectangle b) {
        final int x1 = Math.max(a.x, b.x);
        final int y1 = Math.max(a.y, b.y);
        final int x2 = Math.min(a.x + a.width, b.x + b.width);
        final int y2 = Math.min(a.y + a.height, b.y + b.height);

        if (x2 <= x1 || y2 <= y1) {
            return 0.0f;
        }

        final float intersection = (x2 - x1) * (y2 - y1);
        final float areaA = a.width * a.height;
        final float areaB = b.width * b.height;
        final float union = areaA + areaB - intersection;

        return intersection / union;
    }

    private String extractModel() {
        try {
            final InputStream resourceStream = getClass().getClassLoader()
                .getResourceAsStream("models/yolov8n.onnx");

            if (resourceStream == null) {
                throw new RuntimeException(
                    "Model not found. Please download yolov8n.onnx and place in src/main/resources/models/\n" +
                    "See README for download instructions."
                );
            }

            final File tempFile = File.createTempFile("yolov8n", ".onnx");
            tempFile.deleteOnExit();

            Files.copy(resourceStream, tempFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
            resourceStream.close();

            return tempFile.getAbsolutePath();
        } catch (final Exception e) {
            throw new RuntimeException("Failed to extract model: " + e.getMessage(), e);
        }
    }

    public void close() {
        try {
            if (session != null) session.close();
        } catch (final Exception e) {
            // Ignore
        }
    }
}
