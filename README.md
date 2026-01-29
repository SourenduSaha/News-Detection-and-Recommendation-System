# 📰 News Recommender & Fake News Detection System

## Overview
The rapid spread of online misinformation has made it increasingly difficult for users to distinguish between fake and credible news. While many systems focus only on fake news detection, identifying misinformation alone is insufficient. This project addresses that gap by combining fake news detection with a news recommendation system that actively redirects users toward trustworthy and relevant information.

This repository implements a two-stage machine learning pipeline that first classifies news articles as fake or real and then recommends relevant, verified articles to improve factual understanding and user awareness.

---

## System Architecture
The system consists of two core components:

### 1. Fake News Detection
- Uses a fine-tuned BERT transformer model for binary classification (Fake vs Real).
- Articles are classified using both title and content, leveraging contextual and linguistic representations.
- The model was trained on a diverse dataset collected from multiple news sources.
- Achieved strong performance with:
  - 98.47% accuracy on training data  
  - 96.07% accuracy on validation data  
  - 96.52% accuracy on test data

---

### 2. News Recommendation Engine
Based on the classification outcome:
- If the article is fake, the system retrieves and recommends real, verified articles discussing the same topic.
- If the article is real, the system recommends additional trustworthy articles to broaden the reader’s knowledge.

The recommendation pipeline includes:
- Content-Based Filtering using semantic similarity (GloVe word embeddings)
- User-Based Collaborative Filtering
- Item-Based Collaborative Filtering
- Hybrid Recommendation System combining collaborative signals with contextual and time-based boosts

---

## Key Features
- Semantic similarity using vector embeddings
- Time-based feature engineering (morning, midday, evening, night usage patterns)
- Cold-start mitigation using hybrid recommendation strategies
- Pre-computed embeddings for improved performance and reduced latency
- Scalable design using sparse matrices and chunked data loading

---

## Tech Stack
- Python  
- BERT (Transformers)  
- PyTorch  
- Scikit-learn  
- GloVe Embeddings  
- Pandas, NumPy, Matplotlib  

---

## 📄 Project Report

The full project report and presentation slides are available here:  
[Download PDF – News Detection & Recommendation System](https://drive.google.com/file/d/156jH8nVMPUzGdUtk5Sr0-3Hu1DkivdmH/view?usp=drive_link)
---

## Future Work
- Incorporate real-time news ingestion
- Add source credibility scoring
- Improve personalization with user feedback loops
- Deploy as a web-based application
