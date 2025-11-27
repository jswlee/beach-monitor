# Beach Monitor Inference API

FastAPI service that exposes beach monitoring computer vision models as REST endpoints.

## Features

- **Object Detection**: Detect people and boats using fine-tuned YOLOv8
- **Segmentation**: Classify people as on beach or in water using SegFormer
- **Complete Analysis**: Run both detection and segmentation in one call
- **Snapshot Capture**: Capture fresh images from beach livestream
- **Auto-generated Documentation**: OpenAPI/Swagger docs at `/docs`

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI API Key (for weather analysis)
OPENAI_API_KEY=your-key-here

# YouTube API Key (for livestream capture)
YOUTUBE_API_KEY=your-key-here

# AWS Credentials (for model loading from S3)
AWS_ACCESS_KEY_ID=your-key-here
AWS_SECRET_ACCESS_KEY=your-secret-here

# S3 Model Locations
S3_BUCKET_NAME=beach-detection-model-yolo-finetuned
S3_MODEL_KEY=best.pt

# Segmentation Model (optional, uses defaults if not set)
SEG_S3_BUCKET_NAME=beach-detection-model-yolo-finetuned
SEG_S3_CONFIG_KEY=waterline-model-config.json
SEG_S3_WEIGHTS_KEY=waterline-model.safetensors

# API Configuration (optional)
INFERENCE_API_URL=http://localhost:8000
```

### 3. Run the Server

```bash
# Development mode with auto-reload
uvicorn inference_service:app --reload --host 0.0.0.0 --port 8000

# Or run directly
python inference_service.py
```

### 4. Access API Documentation

Open your browser to:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health Check
```bash
GET /health
```

### Model Information
```bash
GET /models/info
```

### Capture Snapshot
```bash
POST /capture
```
Captures a fresh snapshot from the beach livestream.

**Response:**
```json
{
  "snapshot_url": "/images/youtube_snapshot_20250902_130003.jpg",
  "timestamp": "2025-09-02T13:00:03",
  "processing_time_ms": 1234.5
}
```

### Object Detection
```bash
POST /detect
Content-Type: multipart/form-data

file: <image file>
save_annotated: true (optional)
```

Detects people and boats in an uploaded image.

**Response:**
```json
{
  "people_count": 12,
  "boat_count": 3,
  "activity_level": "moderate",
  "summary": "Detected 12 people and 3 boats. Beach is moderate.",
  "annotated_image_url": "/images/temp_annotated.jpg",
  "processing_time_ms": 567.8
}
```

### Beach/Water Segmentation
```bash
POST /segment
Content-Type: multipart/form-data

file: <image file>
person_boxes: <JSON string of bounding boxes> (optional)
save_segmented: true (optional)
```

Classifies people as on beach or in water.

**Response:**
```json
{
  "beach_count": 8,
  "water_count": 4,
  "other_count": 0,
  "total_count": 12,
  "classifications": [
    {"box_id": 0, "location": "beach", "point": [100, 200]},
    {"box_id": 1, "location": "water", "point": [150, 250]}
  ],
  "segmented_image_url": "/images/temp_segmented.jpg",
  "processing_time_ms": 890.2
}
```

### Complete Analysis
```bash
POST /analyze
Content-Type: multipart/form-data

file: <image file>
```

Runs both detection and segmentation.

**Response:**
```json
{
  "people_count": 12,
  "boat_count": 3,
  "beach_count": 8,
  "water_count": 4,
  "other_count": 0,
  "activity_level": "moderate",
  "summary": "Beach is moderate with 12 people (8 on beach, 4 in water) and 3 boats visible.",
  "annotated_image_url": "/images/temp_annotated.jpg",
  "segmented_image_url": "/images/temp_segmented.jpg",
  "processing_time_ms": 1456.3
}
```

### Get Image
```bash
GET /images/{filename}
```

Serves generated images (annotated, segmented, snapshots).

## Usage Examples

### cURL

```bash
# Capture snapshot
curl -X POST http://localhost:8000/capture

# Analyze an image
curl -X POST -F "file=@beach.jpg" http://localhost:8000/analyze

# Detect objects only
curl -X POST -F "file=@beach.jpg" http://localhost:8000/detect

# Health check
curl http://localhost:8000/health
```

### Python

```python
import requests

# Capture snapshot
response = requests.post("http://localhost:8000/capture")
result = response.json()
print(f"Snapshot captured: {result['snapshot_url']}")

# Analyze an image
with open("beach.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/analyze", files=files)
    result = response.json()
    print(f"People: {result['people_count']}")
    print(f"On beach: {result['beach_count']}, In water: {result['water_count']}")
```

### JavaScript

```javascript
// Capture snapshot
fetch('http://localhost:8000/capture', { method: 'POST' })
  .then(res => res.json())
  .then(data => console.log('Snapshot:', data.snapshot_url));

// Analyze an image
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/analyze', {
  method: 'POST',
  body: formData
})
  .then(res => res.json())
  .then(data => {
    console.log(`People: ${data.people_count}`);
    console.log(`Beach: ${data.beach_count}, Water: ${data.water_count}`);
  });
```

## Using with LangGraph Agent

The API can be used by the LangGraph agent via the tools in `agent/api_tools.py`:

```python
from agent.api_tools import (
    capture_snapshot_api_tool,
    analyze_beach_api_tool,
    detect_objects_api_tool,
    segment_beach_water_api_tool,
    check_api_health
)

# In your agent initialization
self.tools = [
    capture_snapshot_api_tool,
    analyze_beach_api_tool,
    check_api_health,
    # ... other tools
]
```

Set the API URL in your environment:
```bash
export INFERENCE_API_URL=http://localhost:8000
```

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "inference_service:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t beach-monitor-api .
docker run -p 8000:8000 --env-file .env beach-monitor-api
```

### Cloud Platforms

- **AWS Lambda**: Use Mangum adapter for serverless deployment
- **Google Cloud Run**: Deploy as containerized service
- **Heroku**: Use Procfile with uvicorn
- **Railway**: Connect GitHub repo and deploy

## Performance Optimization

### Model Caching

Models are loaded once and reused across requests (lazy loading).

### Image Caching

Snapshots are cached for 5 minutes to reduce API calls to YouTube.

### Async Processing

Use background tasks for cleanup and optional async endpoints for long-running operations.

## Monitoring

### Prometheus Metrics

Add metrics endpoint (see `docs/IMPROVEMENTS.md`):
```bash
GET /metrics
```

### Logging

Logs are written to stdout in JSON format for easy parsing.

## Security

### Rate Limiting

Implement rate limiting to prevent abuse (see `docs/IMPROVEMENTS.md`).

### API Keys

Add API key authentication for production use.

### CORS

Configure CORS origins for production:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Troubleshooting

### Models Not Loading

Ensure AWS credentials are set and S3 bucket is accessible:
```bash
aws s3 ls s3://beach-detection-model-yolo-finetuned/
```

### Snapshot Capture Fails

Check YouTube API key and stream availability:
```bash
curl "https://www.googleapis.com/youtube/v3/search?part=snippet&q=LIVE+24/7+4K+MAUI+LIVE+CAM&type=video&key=YOUR_KEY"
```

### Out of Memory

Reduce image resolution or batch size. Consider using GPU for inference.

## Contributing

See main project README for contribution guidelines.

## License

See main project LICENSE file.
