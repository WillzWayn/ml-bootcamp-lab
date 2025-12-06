# Section 5: Model Design

We will start with a baseline model before moving to complex Deep Learning.

* **Baseline:** Logistic Regression (Fast, interpretable).
* **Advanced:** Fine-tuned BERT (Slower, higher accuracy).

## Implementation

Let's train a simple classifier on our dummy data right now.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Training Data
X_train = [
    "refund my money", "charged too much",  # billing
    "server error 500", "api not responding",  # technical
    "buy new license", "sales inquiry"  # sales
]
y_train = ["billing", "billing", "technical", "technical", "sales", "sales"]

# Build Pipeline
model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())

# Train
model.fit(X_train, y_train)

# Predict
new_tickets = [
    "I am getting a 500 error on the server",
    "I need a refund immediately",
    "Do you have enterprise plans available?"
]

predictions = model.predict(new_tickets)

print("Predictions:")
for ticket, prediction in zip(new_tickets, predictions):
    print(f"  '{ticket}' -> {prediction}")
```

## Model Selection Rationale

**Naive Bayes** for baseline:
- Fast training and inference.
- Works well with TF-IDF features.
- Probabilistic interpretation (confidence scores).

**Advanced alternatives:**
- **Logistic Regression:** Better for high-dimensional text, more stable than NB.
- **SVM:** Strong performance on text classification.
- **Gradient Boosting (XGBoost/LightGBM):** Handles feature interactions.
- **BERT/RoBERTa:** State-of-the-art, but slower and requires GPUs.

---

# Section 6: Evaluation Design

How do we know it works?

## Evaluation Metrics

**Accuracy:** Overall correctness.
$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$

**F1-Score:** Harmonic mean of precision and recall. Crucial because ticket categories are likely imbalanced (e.g., 90% technical, 10% billing).
$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Confusion Matrix:** To see which categories get confused (e.g., confusing "Billing" with "Sales").

## Example Evaluation

```python
from sklearn.metrics import confusion_matrix, classification_report

# Validation data
X_val = [
    "refund request",
    "server down",
    "license upgrade",
    "payment failed",
    "api timeout"
]
y_val = ["billing", "technical", "sales", "billing", "technical"]

# Evaluate
y_pred = model.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nF1 Score (weighted):", f1_score(y_val, y_pred, average='weighted'))
print("\nClassification Report:")
print(classification_report(y_val, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))
```

## Offline vs. Online Evaluation

**Offline:** Test set metrics (backtesting).
- Run model on historical labeled data.
- Measure accuracy, precision, recall, F1.
- Fast feedback loop.

**Online:** A/B testing the model against human agents or a previous model version.
- Route a percentage of tickets to the model.
- Compare routing accuracy and resolution time.
- Measure user satisfaction.

## Success Criteria

✓ **Accuracy > 85%** on validation set.
✓ **F1-Score > 0.80** on minority classes.
✓ **Latency < 200ms** per prediction.
✓ **Throughput ≥ 50 tickets/sec**.
