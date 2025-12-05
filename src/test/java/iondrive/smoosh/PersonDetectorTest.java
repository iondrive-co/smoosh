package iondrive.smoosh;

import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;

import java.awt.Rectangle;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Feature-focused tests for PersonDetector.
 * Tests cover user-facing behavior, not implementation details.
 */
class PersonDetectorTest {

    private PersonDetector detector;

    @BeforeEach
    void setup() {
        // Note: This will fail if model not downloaded. That's intentional - clear error message.
        try {
            detector = new PersonDetector();
        } catch (RuntimeException e) {
            if (e.getMessage().contains("Model not found")) {
                fail("Model not found. Run: ./download_model.sh");
            }
            throw e;
        }
    }

    @AfterEach
    void cleanup() {
        if (detector != null) {
            detector.close();
        }
    }

    @Test
    @DisplayName("Should detect person in test image")
    void shouldDetectPersonInTestImage() {
        final String imagePath = getClass().getClassLoader()
            .getResource("test-images/person1.jpg").getPath();

        final List<PersonDetector.Detection> detections = detector.detectPeople(imagePath);

        assertFalse(detections.isEmpty(), "Should detect at least one person");
        assertTrue(detections.get(0).confidence > 0.25f,
            "Detection confidence should exceed threshold");
    }

    @Test
    @DisplayName("Should detect multiple people in group photo")
    void shouldDetectMultiplePeople() {
        final String imagePath = getClass().getClassLoader()
            .getResource("test-images/group.jpg").getPath();

        final List<PersonDetector.Detection> detections = detector.detectPeople(imagePath);

        assertTrue(detections.size() >= 2,
            "Should detect multiple people in group photo, found: " + detections.size());
    }

    @Test
    @DisplayName("Should return valid bounding boxes")
    void shouldReturnValidBoundingBoxes() {
        final String imagePath = getClass().getClassLoader()
            .getResource("test-images/person2.jpg").getPath();

        final List<PersonDetector.Detection> detections = detector.detectPeople(imagePath);

        for (PersonDetector.Detection detection : detections) {
            assertTrue(detection.bbox.width > 0, "Bbox width should be positive");
            assertTrue(detection.bbox.height > 0, "Bbox height should be positive");
            assertTrue(detection.bbox.x >= 0, "Bbox x should be non-negative");
            assertTrue(detection.bbox.y >= 0, "Bbox y should be non-negative");
        }
    }

    @Test
    @DisplayName("Should return confidence scores in valid range")
    void shouldReturnValidConfidenceScores() {
        final String imagePath = getClass().getClassLoader()
            .getResource("test-images/person1.jpg").getPath();

        final List<PersonDetector.Detection> detections = detector.detectPeople(imagePath);

        for (PersonDetector.Detection detection : detections) {
            assertTrue(detection.confidence >= 0.0f && detection.confidence <= 1.0f,
                "Confidence should be between 0 and 1, got: " + detection.confidence);
        }
    }

    @Test
    @DisplayName("Should handle empty or uniform images gracefully")
    void shouldHandleEmptyImage(@TempDir Path tempDir) {
        final String imagePath = CreateTestImage.createUniformImage(
            tempDir, "empty.jpg", 640, 480);

        final List<PersonDetector.Detection> detections = detector.detectPeople(imagePath);

        // Should not crash, may return empty list or no valid detections
        assertNotNull(detections, "Should return non-null list");
    }

    @Test
    @DisplayName("Should throw exception for invalid image path")
    void shouldThrowExceptionForInvalidPath() {
        final RuntimeException exception = assertThrows(RuntimeException.class, () ->
            detector.detectPeople("/nonexistent/image.jpg")
        );

        assertTrue(exception.getMessage().contains("Failed to load image"),
            "Should provide clear error message");
    }

    @Test
    @DisplayName("detectLargestPerson should return largest detection")
    void detectLargestPersonShouldReturnLargest() {
        final String imagePath = getClass().getClassLoader()
            .getResource("test-images/group.jpg").getPath();

        final Rectangle largest = detector.detectLargestPerson(imagePath);

        assertNotNull(largest, "Should return non-null rectangle");
        assertTrue(largest.width > 0 && largest.height > 0,
            "Should have positive dimensions");
    }

    @Test
    @DisplayName("detectLargestPerson should return full image if no person found")
    void detectLargestPersonShouldReturnFullImageWhenNoPerson(@TempDir Path tempDir) {
        final String imagePath = CreateTestImage.createUniformImage(
            tempDir, "no_person.jpg", 800, 600);

        final Rectangle result = detector.detectLargestPerson(imagePath);

        // Should return full image dimensions as fallback
        assertEquals(800, result.width, "Should return full image width");
        assertEquals(600, result.height, "Should return full image height");
    }

    @Test
    @DisplayName("Should be reusable for multiple detections")
    void shouldBeReusableForMultipleDetections() {
        final String imagePath = getClass().getClassLoader()
            .getResource("test-images/person1.jpg").getPath();

        final List<PersonDetector.Detection> first = detector.detectPeople(imagePath);
        final List<PersonDetector.Detection> second = detector.detectPeople(imagePath);

        assertEquals(first.size(), second.size(),
            "Should return consistent results across multiple calls");
    }

    @Test
    @DisplayName("Detection toString should be readable")
    void detectionToStringShouldBeReadable() {
        final String imagePath = getClass().getClassLoader()
            .getResource("test-images/person1.jpg").getPath();

        final List<PersonDetector.Detection> detections = detector.detectPeople(imagePath);

        if (!detections.isEmpty()) {
            final String str = detections.get(0).toString();
            assertTrue(str.contains("Person"), "toString should contain class name");
            assertTrue(str.contains("bbox"), "toString should contain bbox info");
            assertTrue(str.contains("conf"), "toString should contain confidence");
        }
    }
}
