# Beach Monitor Architecture

## Data Flow

```
YouTube Livestream
    ↓
Snapshot Capture (beach_cv_tool/capture.py)
    ↓
Beach Image (1920x1080 JPEG)
    ↓
Fine-tuned YOLO Model (beach_cv_tool/detection.py)
    ↓
Bounding Boxes: [person1, person2, ..., boat1, ...]
    ↓
Location Classifier (beach_cv_tool/location_classifier.py)
    ↓
GPT-4V analyzes ONLY the detected people boxes
    ↓
Classifications: [beach, water, beach, unclear, ...]
    ↓
Final Counts:
  - Total people: 10
  - On beach: 6
  - In water: 4
  - Boats: 2
```

## Key Components

### 1. Fine-tuned YOLO Model (Your Model)
- **Purpose**: Detect people and boats
- **Input**: Beach image
- **Output**: Bounding boxes with class labels (person/boat)
- **Cost**: Free (runs locally or on your infrastructure)
- **Speed**: Fast (~100ms per image)

### 2. Location Classifier (GPT-4V)
- **Purpose**: Classify detected people as beach/water
- **Input**: 
  - Original image
  - Bounding boxes from YOLO (only people, not boats)
- **Output**: Classification for each person box
- **Cost**: ~$0.01 per image
- **Speed**: 2-5 seconds per image

### 3. Why This Architecture?

**Efficient:**
- YOLO does the heavy lifting (detection)
- VLM only does classification (easier task)
- No redundant object detection

**Accurate:**
- YOLO is fine-tuned on your beach data
- VLM excels at spatial reasoning (beach vs water)
- Numbers always match (same boxes)

**Cost-effective:**
- Only pay for VLM when people are detected
- No VLM cost if beach is empty
- YOLO runs for free

## Example Flow

### Input Image
Beach scene with 10 people and 2 boats

### Step 1: YOLO Detection
```python
detector = BeachDetector()
results = detector.detect_objects(image)
# Returns: 10 person boxes, 2 boat boxes
```

### Step 2: Extract Person Boxes
```python
person_boxes = [
    {'xyxy': [100, 200, 150, 400]},  # Person 1
    {'xyxy': [300, 250, 350, 450]},  # Person 2
    # ... 8 more
]
```

### Step 3: VLM Classification
```python
classifier = LocationClassifier()
result = classifier.classify_locations(image, person_boxes)
# Returns:
# {
#   'beach_count': 6,
#   'water_count': 4,
#   'classifications': [
#     {'box_id': 0, 'location': 'beach'},
#     {'box_id': 1, 'location': 'water'},
#     ...
#   ]
# }
```

### Step 4: Final Output
```
"Beach is moderate with 10 people (6 on beach, 4 in water) and 2 boats visible."
```

## Cost Analysis

### Per Query Costs
- **YOLO Detection**: $0.00 (free)
- **VLM Classification**: $0.01 (only if people detected)
- **Total**: ~$0.01 per query with people

### Monthly Estimates
- **100 queries/day**: ~$30/month
- **500 queries/day**: ~$150/month
- **1000 queries/day**: ~$300/month

### Optimization Tips
1. **Cache results** for 5 minutes to reduce API calls
2. **Disable classification** during off-hours (night)
3. **Skip classification** if people count < 3 (not useful)
4. **Batch queries** if multiple users ask simultaneously

## Error Handling

### YOLO Fails
- Return empty counts
- No VLM call (saves money)

### VLM Fails
- Return total people count
- Set beach/water to 0
- Continue operation gracefully

### Both Fail
- Return cached result if available
- Show error message to user

## Performance

### Typical Response Times
- **YOLO Detection**: 100-200ms
- **VLM Classification**: 2-5 seconds
- **Total**: 2-5 seconds

### Bottlenecks
- VLM API call is the slowest part
- Consider async processing for better UX
- Show loading spinner during VLM call

## Future Enhancements

### Short-term (MVP+)
- [ ] Cache results for 5 minutes
- [ ] Add confidence scores to classifications
- [ ] Track trends over time

### Medium-term
- [ ] Train a lightweight classifier (faster, cheaper)
- [ ] Add more location categories (surfing, kayaking)
- [ ] Detect water conditions (waves, calm)

### Long-term
- [ ] Real-time video analysis
- [ ] Predictive modeling (forecast busy times)
- [ ] Multi-camera support
