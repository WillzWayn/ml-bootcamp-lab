# Section 7: Deployment Design

Once the model is trained, it must be accessible in a production environment.

## Serving Strategy

We will wrap the model in a **REST API** using a container.

### 1. Dockerfile

We define the environment to ensure reproducibility across dev, staging, and production.

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**requirements.txt:**
```
fastapi==0.104.1
uvicorn==0.24.0
scikit-learn==1.3.2
pydantic==2.5.0
joblib==1.3.2
```

### 2. API Interface (FastAPI)

The code for the inference endpoint would look like this:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from typing import Optional

app = FastAPI(title="Ticket Triage API", version="1.0")

# Load the model at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = joblib.load("models/triage_model.pkl")

class TicketRequest(BaseModel):
    ticket_id: str
    subject: str
    body: str

class PredictionResponse(BaseModel):
    ticket_id: str
    category: str
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TicketRequest):
    """
    Classify a support ticket into a category.
    
    Args:
        request: Ticket details (subject + body)
    
    Returns:
        PredictionResponse with predicted category and confidence
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Combine subject and body for prediction
    text = f"{request.subject} {request.body}"
    
    try:
        # Get prediction and confidence
        prediction = model.predict([text])[0]
        confidence = max(model.predict_proba([text])[0])
        
        return PredictionResponse(
            ticket_id=request.ticket_id,
            category=prediction,
            confidence=float(confidence)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint for orchestrators."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/")
async def root():
    """Root endpoint with API documentation."""
    return {
        "api": "Ticket Triage System",
        "version": "1.0",
        "endpoints": {
            "predict": "POST /predict - Classify a ticket",
            "health": "GET /health - Health check",
            "docs": "GET /docs - Interactive API docs (Swagger UI)"
        }
    }
```

### 3. Deployment Architecture

```
┌─────────────────────────────────────────┐
│         Load Balancer (Nginx)           │
└────────────────┬────────────────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───▼──┐    ┌───▼──┐    ┌───▼──┐
│ Pod 1│    │ Pod 2│    │ Pod 3│  (Kubernetes)
│ API  │    │ API  │    │ API  │
│ v1.0 │    │ v1.0 │    │ v1.0 │
└──────┘    └──────┘    └──────┘
    │            │            │
    └────────────┼────────────┘
                 │
        ┌────────▼────────┐
        │ Model Registry  │
        │    (MLflow)     │
        └─────────────────┘
```

**Key Components:**
- **Load Balancer:** Distributes requests.
- **API Pods:** Stateless FastAPI containers.
- **Model Registry:** Centralized model versioning (MLflow).

### 4. Containerization & Orchestration

To run locally with Docker:

```bash
# Build image
docker build -t ticket-triage-api:1.0 .

# Run container
docker run -p 8000:8000 ticket-triage-api:1.0
```

For production (Kubernetes):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ticket-triage-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ticket-triage-api
  template:
    metadata:
      labels:
        app: ticket-triage-api
    spec:
      containers:
      - name: api
        image: ticket-triage-api:1.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
          limits:
            cpu: 1000m
            memory: 1Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
```

---

## 5. Monitoring & Observability

Once deployed, we must monitor:

### Latency Monitoring

```python
import time
from prometheus_client import Histogram

request_duration = Histogram('request_duration_seconds', 'Request latency')

@app.post("/predict")
async def predict(request: TicketRequest):
    with request_duration.time():
        # ... prediction logic
        pass
```

### Data Drift Detection

**Problem:** Language changes over time (new products, seasonal support spikes).

**Solution:** Monitor word distribution in incoming tickets vs. training data.

```python
from scipy.spatial.distance import jensenshannon

def check_data_drift(new_texts, reference_distribution):
    """
    Compute Jensen-Shannon divergence between incoming data
    and training data distributions.
    """
    new_dist = compute_feature_distribution(new_texts)
    drift_score = jensenshannon(new_dist, reference_distribution)
    
    if drift_score > DRIFT_THRESHOLD:
        alert("Data drift detected! Retrain model.")
    
    return drift_score
```

### Key Metrics to Monitor

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Latency (p99) | < 200ms | > 500ms |
| Throughput | 50 req/sec | < 30 req/sec |
| Accuracy (online) | > 85% | < 75% |
| Error Rate | < 1% | > 5% |
| Data Drift | - | JSD > 0.15 |

---

## Summary: Zero to Hero! ✅

You have designed a complete AI system:

✅ **Requirements:** Auto-triage tickets with >85% accuracy.
✅ **Architecture:** API Gateway → Model Service → Predictions.
✅ **Data Pipeline:** Ingestion → Cleaning → Feature Engineering.
✅ **Modeling:** TF-IDF + Naive Bayes baseline → BERT advanced.
✅ **Evaluation:** Accuracy, F1, Confusion Matrix, Online A/B testing.
✅ **Deployment:** Docker → FastAPI → Kubernetes.
✅ **Monitoring:** Latency, data drift, model accuracy tracking.

### Next Steps

1. **Commit and Push:** Save these files to your GitHub repository.
   ```bash
   git add .
   git commit -m "Add AI System Design: Ticket Triage course"
   git push origin main
   ```

2. **Run Locally:** Start your Labspace to preview the course:
   ```bash
   docker compose -f docker-compose.yml -f .labspace/compose.override.yaml up
   ```

3. **View:** Open `http://localhost:3000` and walk through your new AI System Design course!

4. **Implement:** Use the provided code snippets to build a real triage system with your own data.

Happy learning! 🚀
