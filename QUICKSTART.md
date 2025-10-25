# Quick Start Guide

## New Feature: Beach vs Water Classification 🏖️🌊

The beach monitor now separates people counts into:
- **On Beach**: People on sand/shore
- **In Water**: People swimming/wading
- **Unclear**: Ambiguous positions

This uses GPT-4V to analyze bounding boxes and classify locations.

---

## Local Development

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Required
OPENAI_API_KEY=sk-your-openai-api-key
YOUTUBE_API_KEY=your-youtube-api-key

# For model loading from S3
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
S3_BUCKET_NAME=your-bucket-name
S3_MODEL_KEY=path/to/model.pt
```

### 3. Run the App

```bash
python main.py
```

Or directly:

```bash
streamlit run ui/chat.py
```

The app will open at `http://localhost:8501`

---

## Testing Location Classification

### Test the classifier standalone:

```python
from beach_cv_tool.location_classifier import LocationClassifier
import cv2

# Initialize
classifier = LocationClassifier()

# Load an image
image = cv2.imread("path/to/beach_image.jpg")

# Define bounding boxes (from YOLO detection)
boxes = [
    {'xyxy': [100, 100, 200, 300]},  # Person 1
    {'xyxy': [500, 200, 600, 400]},  # Person 2
]

# Classify
result = classifier.classify_locations(image, boxes)
print(f"Beach: {result['beach_count']}, Water: {result['water_count']}")
```

### Test the full pipeline:

```python
from beach_cv_tool.detection import BeachDetector

# Initialize with location classification enabled (default)
detector = BeachDetector(enable_location_classification=True)

# Analyze an image
result = detector.analyze_beach_activity("path/to/image.jpg")
print(result['summary'])
# Output: "Beach is moderate with 10 people (6 on beach, 4 in water) and 2 boats visible."
```

### Disable location classification (faster, no GPT-4V cost):

```python
detector = BeachDetector(enable_location_classification=False)
```

---

## Example Queries

Try these in the chat UI:

- "How many people are on the beach vs in the water?"
- "Are more people swimming or on the sand?"
- "What's the current beach activity?"
- "Show me the annotated image"

---

## Cost Optimization

### GPT-4V Usage
- **Cost**: ~$0.01 per image analysis
- **When it runs**: Only when people are detected
- **Disable if needed**: Set `enable_location_classification=False`

### Caching Strategy
Consider caching results for a few minutes to reduce API calls:

```python
import time
from functools import lru_cache

@lru_cache(maxsize=10)
def cached_analysis(image_path, timestamp_bucket):
    # timestamp_bucket = int(time.time() / 300)  # 5-minute buckets
    return detector.analyze_beach_activity(image_path)
```

---

## Troubleshooting

### "OpenAI API key not found"
- Ensure `OPENAI_API_KEY` is in your `.env` file
- Check that `.env` is in the project root
- Restart the app after adding the key

### "Location classifier failed to initialize"
- The app will continue without location classification
- Check OpenAI API key is valid
- Verify internet connection

### "Model not found in S3"
- Verify `S3_BUCKET_NAME` and `S3_MODEL_KEY` are correct
- Check AWS credentials have S3 read permissions
- Ensure the model file exists at the specified S3 path

### Slow performance
- GPT-4V calls take 2-5 seconds
- Consider disabling location classification for faster responses
- Use caching for repeated queries

---

## Architecture

```
User Query
    ↓
Streamlit UI (ui/chat.py)
    ↓
LangGraph Agent (agent/graph.py)
    ↓
Beach Monitoring Tool (agent/tools.py)
    ↓
Beach Detector (beach_cv_tool/detection.py)
    ↓
├─ YOLO Detection (person/boat bounding boxes)
    ↓
└─ Location Classifier (beach_cv_tool/location_classifier.py)
       ↓
    GPT-4V (classifies each person as beach/water)
```

---

## Next Steps

1. **Test locally** with real beach images
2. **Deploy** using `DEPLOYMENT.md` guide
3. **Monitor costs** in OpenAI dashboard
4. **Optimize** based on usage patterns

For deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)
