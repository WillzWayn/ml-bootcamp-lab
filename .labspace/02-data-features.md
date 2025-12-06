# Section 3: Data Engineering Design

Data is the fuel for our system. We assume we have a dataset of historical tickets labeled by humans.

## Data Pipeline Steps
1. **Ingestion:** Load CSV/SQL dump of past tickets.
2. **Cleaning:** Remove PII (Personally Identifiable Information) like emails and credit card numbers.
3. **Splitting:** 80% Train, 10% Validation, 10% Test.

Let's simulate loading and viewing some raw data:

```python
import pandas as pd

# Simulating raw data
data = {
    'ticket_id': [101, 102, 103],
    'text': [
        "My credit card was charged twice! Refund needed.",
        "How do I install the python SDK on Windows?",
        "I want to upgrade my enterprise plan."
    ],
    'category': ['billing', 'technical', 'sales']
}

df = pd.DataFrame(data)
print("Raw Data Sample:")
print(df)
```

---

# Section 4: Feature Engineering Design

Raw text cannot be understood by machines. We must convert it into numerical vectors.

## Vectorization Techniques

**Text Cleaning:** Lowercasing, removing punctuation.

**Vectorization:** We will use TF-IDF (Term Frequency-Inverse Document Frequency) for the baseline, as it is fast and effective for keyword-heavy tickets.

Try this simple feature engineering pipeline:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Define the vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# 2. Fit and transform the text
texts = [
    "My credit card was charged twice! Refund needed.",
    "How do I install the python SDK on Windows?",
    "I want to upgrade my enterprise plan."
]
vectors = vectorizer.fit_transform(texts)

# 3. View the token map
print("Vocabulary size:", len(vectorizer.get_feature_names_out()))
print("Vector Shape:", vectors.shape)
print("Sample feature names:", vectorizer.get_feature_names_out()[:10])
```

## Why TF-IDF?

- **Fast:** Vectorization is O(n) and vectorized inference is instant.
- **Interpretable:** Easy to understand which words matter.
- **Effective:** Works well on keyword-heavy support tickets.
- **Scalable:** Can handle millions of documents.

For higher accuracy, we could later upgrade to:
- **Word2Vec / FastText:** Semantic embeddings.
- **BERT / RoBERTa:** Pre-trained transformer models (slower, more accurate).
