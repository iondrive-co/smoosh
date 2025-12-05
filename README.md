# Smoosh - Modern Person Detection for Java

Simple, accurate person detection using YOLOv8 via ONNX Runtime.

## Quick Start

### 1. Model Setup

```bash
# Model downloads automatically when you build
./gradlew build
```
or
```bash
chmod +x download_model.sh
./download_model.sh
```
The model (~6MB) is bundled in the JAR, so end users don't need to download anything.

### 2. Use as Library

```java
import iondrive.smoosh.PersonDetector;
import java.util.List;

PersonDetector detector = new PersonDetector();
List<PersonDetector.Detection> people = detector.detectPeople("photo.jpg");

for (PersonDetector.Detection person : people) {
    System.out.println("Found person at: " + person.bbox);
    System.out.println("Confidence: " + person.confidence);
}

detector.close();
```

### 3. Visual Demo

```bash
./gradlew run --args="photo.jpg"
```

This displays the image with bounding boxes around detected people.
