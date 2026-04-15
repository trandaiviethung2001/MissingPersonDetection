# Missing Person Detection System

## Architecture

```
Input Video (crowd footage)
        |
        v
  +------------------+
  | YOLOv8 Detection |  --> Detect all PERSONS in frame
  +------------------+
        |
        v (crop each person)
  +---------------------+
  | InsightFace         |  --> Find FACES + generate 512-d EMBEDDING
  | (ArcFace ResNet100) |
  +---------------------+
        |
        v (cosine similarity)
  +------------------+
  | Face Recognition |  --> Match against missing person DATABASE
  +------------------+
        |
        v
  ALERT if match!
```

## Installation

```bash
cd MissingPersonDetection
pip install -r requirements.txt
```

Key dependencies:
- `ultralytics` - YOLOv8 person detection
- `insightface` - ArcFace face detection + recognition (512-d embeddings)
- `onnxruntime` - Runtime for InsightFace models
- `opencv-python` - Image/video processing
- `scikit-learn` - SVM/KNN classifier training (optional)

## Usage

### Step 1: Prepare missing person photos

Create a folder for each person in `missing_persons_db/`:

```
missing_persons_db/
  person_001/
    name.txt          # Person's name, e.g. "John Doe"
    photo1.jpg         # Clear frontal face photo
    photo2.jpg         # Side angle photo
    photo3.jpg         # Different lighting conditions
  person_002/
    name.txt
    photo1.jpg
```

Photo guidelines:
- 3-5 photos per person from different angles and lighting
- Face should be clear and unobstructed
- Minimum 100x100 pixels for face region

### Step 2: Build embeddings database

Open and run the notebook:
```bash
jupyter notebook train_face_recognition.ipynb
```

The notebook will:
1. Load and display images from database
2. Generate ArcFace 512-d embeddings for each person
3. Evaluate embedding quality (intra/inter-person cosine similarity)
4. Find optimal recognition threshold (ROC curve)
5. Save `embeddings.pkl`
6. (Optional) Train SVM/KNN classifier

### Step 3: Run detection on video

```bash
# Basic - display results on screen
python detect_missing_person.py --video path/to/crowd_video.mp4

# Save output video
python detect_missing_person.py --video path/to/video.mp4 --output output/result.mp4

# Custom parameters
python detect_missing_person.py \
    --video path/to/video.mp4 \
    --output output/result.mp4 \
    --threshold 0.5 \
    --skip 3 \
    --no-display
```

### Command-line arguments

| Argument      | Description                                    | Default  |
|---------------|------------------------------------------------|----------|
| `--video`     | Input video path (required)                    | -        |
| `--output`    | Output video path                              | None     |
| `--db`        | Path to embeddings.pkl                         | config   |
| `--threshold` | Cosine similarity threshold (higher = stricter)| 0.4      |
| `--skip`      | Process every N frames                         | 5        |
| `--no-display`| Disable video display window                   | False    |

## Project Structure

```
MissingPersonDetection/
  config.py                    # System configuration
  requirements.txt             # Dependencies
  detect_missing_person.py     # Main detection pipeline
  train_face_recognition.ipynb # Embeddings & training notebook
  missing_persons_db/          # Missing person photos
    person_001/
    person_002/
    embeddings.pkl             # Embeddings file (auto-generated)
  utils/
    __init__.py
    person_detector.py         # YOLO person detection
    face_detector.py           # InsightFace face detection
    face_recognizer.py         # ArcFace recognition & matching
  models/                      # Model weights (SVM/KNN)
  output/                      # Output results
```

## Tuning

Edit `config.py`:

- `RECOGNITION_THRESHOLD`: Increase (e.g. 0.5) to reduce false positives, decrease (e.g. 0.3) to catch more matches
- `FRAME_SKIP`: Decrease to process more frames (slower), increase for speed
- `INSIGHTFACE_MODEL`: "buffalo_l" (accurate) or "buffalo_s" (fast)
- `YOLO_CONFIDENCE_THRESHOLD`: Increase to only detect clear persons
- `FACE_DETECTION_THRESHOLD`: Face detection confidence threshold

## Model Performance

Training was performed using the `train_face_recognition.ipynb` notebook on a database of **3 persons** (Morgan Freeman, Keanu Reeves, Scarlett Johansson) with **3 photos each** (9 total embeddings).

### Database Overview

![Database Overview](output/database_overview.png)

### Embedding Generation

| Metric | Value |
|--------|-------|
| Embedding model | ArcFace ResNet100 (buffalo_l) |
| Embedding dimensions | 512-d |
| Total persons | 3 |
| Total embeddings | 9 |
| Face detection success rate | 100% (9/9) |
| Embeddings file size | 18.6 KB |

### Embedding Quality (Cosine Similarity)

**Intra-person similarity** (same person, higher is better):

| Person | Avg Similarity | Min Similarity | Pairs |
|--------|---------------|----------------|-------|
| Morgan Freeman | 0.552 | 0.465 | 3 |
| Keanu Reeves | 0.245 | 0.038 | 3 |
| Scarlett Johansson | 0.643 | 0.616 | 3 |
| **Overall** | **0.480** | **0.038** | **9** |

**Inter-person similarity** (different persons, lower is better):

| Pair | Avg Similarity | Max Similarity |
|------|---------------|----------------|
| Morgan Freeman vs Keanu Reeves | -0.007 | 0.048 |
| Morgan Freeman vs Scarlett Johansson | -0.005 | 0.046 |
| Keanu Reeves vs Scarlett Johansson | -0.028 | 0.035 |
| **Overall** | **-0.013** | **0.048** |

> **Note:** Inter-person similarity is near zero across all pairs, indicating excellent separation between identities. Keanu Reeves has lower intra-person similarity due to significant appearance variation across photos (different years, angles, facial hair).

### Similarity Distribution

![Cosine Similarity Distribution](output/similarity_distribution.png)

The histogram shows clear separation between intra-person (blue) and inter-person (red) similarity distributions. The green dashed line marks the recognition threshold (0.4).

### Threshold Tuning & ROC Curve

![Threshold Tuning](output/threshold_tuning.png)

| Metric | Value |
|--------|-------|
| Optimal threshold | 0.10 |
| Best accuracy at optimal threshold | 94.4% |
| Config threshold | 0.40 |

The left plot shows accuracy, TPR, precision, and F1 score across different threshold values. The right plot shows the ROC curve.

### Classifier Performance (Leave-One-Out Cross-Validation)

| Classifier | Accuracy | Std Dev | Samples | Classes |
|------------|----------|---------|---------|---------|
| **SVM** (RBF kernel) | 88.9% | ±31.4% | 9 | 3 |
| **KNN** (k=3, distance-weighted) | 100.0% | ±0.0% | 9 | 3 |

> **Note:** High variance in SVM is expected with only 9 samples and leave-one-out cross-validation. KNN achieves perfect accuracy due to the well-separated embedding clusters. With more training data, both classifiers would stabilize.

### Key Takeaways

- **Inter-person separation is excellent**: max inter-person similarity (0.048) is far below the recognition threshold (0.4), meaning false positives are unlikely
- **Intra-person consistency varies**: Scarlett Johansson's embeddings are consistent (avg 0.643), while Keanu Reeves shows more variation (avg 0.245) due to appearance changes across years
- **Cosine similarity matching** (threshold-based) is the primary recognition method; SVM/KNN classifiers serve as optional alternatives
- Adding more reference photos per person (especially with diverse angles/lighting) improves intra-person similarity and overall robustness

## Comparison: InsightFace vs dlib

| Feature            | InsightFace (ArcFace)      | dlib (face_recognition)  |
|--------------------|----------------------------|--------------------------|
| Embedding          | 512-d                      | 128-d                    |
| Accuracy (LFW)     | 99.77%                     | 99.38%                   |
| macOS ARM install  | pip install (easy)         | Requires building dlib   |
| Model              | ArcFace ResNet100          | dlib ResNet              |
| Comparison method  | Cosine similarity          | L2 distance              |
