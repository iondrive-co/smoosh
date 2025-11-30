# Smoosh

Uses OpenCV to detect regions of interest in images and return a bounding box encompassing them.
Run with ./gradlew runGui or import and call RegionDetector#detectRegionOfInterest

## How It Works

### Edge Detection
1. Converts image to grayscale
2. Applies Gaussian blur to reduce noise
3. Runs Canny edge detection
4. Finds contours and returns bounding box of largest contour

### Saliency Detection
1. Resizes image for faster processing
2. Applies Laplacian operator to find high-frequency regions
3. Thresholds to create binary mask
4. Finds contours and returns bounding box of largest salient region

### Face Detection
1. Converts to grayscale and equalizes histogram
2. Uses Haar Cascade classifier to detect faces
3. Returns bounding box of largest face detected
