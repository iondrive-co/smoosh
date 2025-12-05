package iondrive.smoosh;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.awt.Rectangle;
import java.io.InputStream;
import java.util.*;

public class PersonDetectionBenchmark {

    static class GroundTruth {
        String filename;
        List<Rectangle> persons;

        GroundTruth(final String filename) {
            this.filename = filename;
            this.persons = new ArrayList<>();
        }
    }

    static class BenchmarkMetrics {
        int truePositives = 0;
        int falsePositives = 0;
        int falseNegatives = 0;
        double totalIoU = 0.0;
        int detectionCount = 0;

        double getPrecision() {
            final int total = truePositives + falsePositives;
            return total == 0 ? 0.0 : (double) truePositives / total;
        }

        double getRecall() {
            final int total = truePositives + falseNegatives;
            return total == 0 ? 0.0 : (double) truePositives / total;
        }

        double getF1Score() {
            final double precision = getPrecision();
            final double recall = getRecall();
            return precision + recall == 0 ? 0.0 : 2 * (precision * recall) / (precision + recall);
        }

        double getAverageIoU() {
            return detectionCount == 0 ? 0.0 : totalIoU / detectionCount;
        }
    }

    public static void main(final String[] args) throws Exception {
        final List<GroundTruth> groundTruths = loadGroundTruth();

        System.out.println("Person Detection Benchmark - YOLOv8");
        System.out.println("=".repeat(80));
        System.out.println();

        try (final PersonDetector detector = new PersonDetector()) {
            final BenchmarkMetrics metrics = benchmarkDetection(groundTruths, detector);

            System.out.println("Results:");
            System.out.println("-".repeat(80));
            System.out.printf("  Precision:    %.2f%%\n", metrics.getPrecision() * 100);
            System.out.printf("  Recall:       %.2f%%\n", metrics.getRecall() * 100);
            System.out.printf("  F1 Score:     %.2f%%\n", metrics.getF1Score() * 100);
            System.out.printf("  Average IoU:  %.2f%%\n", metrics.getAverageIoU() * 100);
            System.out.printf("  TP/FP/FN:     %d/%d/%d\n",
                metrics.truePositives, metrics.falsePositives, metrics.falseNegatives);
            System.out.println();

            final String grade = getPerformanceGrade(metrics.getF1Score());
            System.out.println("Performance Grade: " + grade);
            System.out.println("=".repeat(80));
        }
    }

    private static BenchmarkMetrics benchmarkDetection(final List<GroundTruth> groundTruths,
                                                       final PersonDetector detector) {
        final BenchmarkMetrics metrics = new BenchmarkMetrics();

        for (final GroundTruth gt : groundTruths) {
            final String imagePath = PersonDetectionBenchmark.class.getClassLoader()
                .getResource("test-images/" + gt.filename).getPath();

            final List<PersonDetector.Detection> detections = detector.detectPeople(imagePath);

            final Set<Integer> matchedGroundTruth = new HashSet<>();

            for (final PersonDetector.Detection detection : detections) {
                double maxIoU = 0.0;
                int bestMatch = -1;

                for (int j = 0; j < gt.persons.size(); j++) {
                    if (matchedGroundTruth.contains(j)) continue;

                    final Rectangle gtRect = gt.persons.get(j);
                    final double iou = calculateIoU(detection.bbox, gtRect);

                    if (iou > maxIoU) {
                        maxIoU = iou;
                        bestMatch = j;
                    }
                }

                if (maxIoU >= 0.5) {
                    metrics.truePositives++;
                    metrics.totalIoU += maxIoU;
                    metrics.detectionCount++;
                    matchedGroundTruth.add(bestMatch);
                } else {
                    metrics.falsePositives++;
                }
            }

            metrics.falseNegatives += gt.persons.size() - matchedGroundTruth.size();
        }

        return metrics;
    }

    private static double calculateIoU(final Rectangle a, final Rectangle b) {
        final int x1 = Math.max(a.x, b.x);
        final int y1 = Math.max(a.y, b.y);
        final int x2 = Math.min(a.x + a.width, b.x + b.width);
        final int y2 = Math.min(a.y + a.height, b.y + b.height);

        if (x2 <= x1 || y2 <= y1) {
            return 0.0;
        }

        final double intersectionArea = (x2 - x1) * (y2 - y1);
        final double areaA = a.width * a.height;
        final double areaB = b.width * b.height;
        final double unionArea = areaA + areaB - intersectionArea;

        return intersectionArea / unionArea;
    }

    private static String getPerformanceGrade(final double f1Score) {
        if (f1Score >= 0.8) return "A (Excellent - Production Ready)";
        if (f1Score >= 0.6) return "B (Good - Usable)";
        if (f1Score >= 0.4) return "C (Fair - Needs Improvement)";
        if (f1Score >= 0.2) return "D (Poor)";
        return "F (Failed - Not Usable)";
    }

    private static List<GroundTruth> loadGroundTruth() throws Exception {
        final ObjectMapper mapper = new ObjectMapper();
        final InputStream is = PersonDetectionBenchmark.class.getClassLoader()
            .getResourceAsStream("test-images/ground_truth.json");

        final JsonNode root = mapper.readTree(is);
        final JsonNode annotations = root.get("annotations");

        final List<GroundTruth> groundTruths = new ArrayList<>();

        for (final JsonNode annotation : annotations) {
            final String filename = annotation.get("filename").asText();
            final GroundTruth gt = new GroundTruth(filename);

            for (final JsonNode person : annotation.get("persons")) {
                final JsonNode bbox = person.get("bbox");
                final Rectangle rect = new Rectangle(
                    bbox.get("x").asInt(),
                    bbox.get("y").asInt(),
                    bbox.get("width").asInt(),
                    bbox.get("height").asInt()
                );
                gt.persons.add(rect);
            }

            groundTruths.add(gt);
        }

        return groundTruths;
    }
}
