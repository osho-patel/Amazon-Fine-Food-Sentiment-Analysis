# Amazon Fine Food Reviews Sentiment Analysis Using LSTM

## Project Overview

This project implements a **Deep Learning-based Sentiment Analysis System** to classify Amazon Fine Food customer reviews as **Positive or Negative** using Natural Language Processing (NLP) and Long Short-Term Memory (LSTM) networks.

The project covers the complete machine learning pipeline, including exploratory data analysis, text preprocessing, tokenization, sequence padding, model training, evaluation, and sentiment prediction on unseen reviews.

## Dataset

**Source:** [Amazon Fine Food Reviews — Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

The dataset contains **568,454 customer reviews** with information such as product IDs, customer ratings, review text, and helpfulness scores.

For sentiment classification, the `Text` and `Score` columns were selected.

### Sentiment Labeling

| Customer Rating | Sentiment | Label |
|---|---|---|
| 1–2 Stars | Negative | 0 |
| 3 Stars | Neutral (Excluded) | — |
| 4–5 Stars | Positive | 1 |

After removing duplicate review-rating pairs and excluding neutral reviews, the final dataset contained **363,903 reviews**.

### Dataset Splitting

| Dataset | Percentage | Samples |
|---|---|---|
| Training | 70% | 254,732 |
| Validation | 15% | 54,585 |
| Testing | 15% | 54,586 |

Stratified sampling was used to preserve class proportions across the datasets.

## Technologies Used

**Programming Language:** Python

**Libraries and Frameworks:**
- TensorFlow and Keras
- NumPy and Pandas
- NLTK
- Scikit-learn
- Matplotlib and Seaborn

**Development Environment:** Google Colab (GPU)

**Dataset Access:** KaggleHub

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

- Analyzed dataset structure, missing values, duplicates, and rating distribution.
- Examined review lengths and class distribution.
- Visualized customer ratings and sentiment distributions.
- Removed duplicate review-rating pairs and excluded neutral ratings.

### 2. Text Preprocessing

Applied NLP preprocessing techniques to prepare review text for deep learning:

- **Lowercasing:** Converted all text to lowercase.
- **Text Cleaning:** Removed HTML tags, URLs, special characters, and unwanted numbers.
- **Stopword Removal:** Removed common English stopwords while preserving important negation words.
- **Stemming:** Applied Porter Stemmer to reduce words to their root forms.
- **Tokenization:** Converted cleaned text into numerical sequences.
- **Padding:** Standardized sequences to a maximum length of 51 tokens.

The tokenizer was fitted only on the training dataset to avoid vocabulary leakage.

### 3. LSTM Model Architecture

The sentiment classifier was developed using TensorFlow and Keras.

| Layer | Configuration |
|---|---|
| Input | Sequence length: 51 |
| Embedding | Vocabulary: 10,000; Dimension: 64 |
| LSTM | 128 units |
| Dropout | 0.4 |
| Dense | 64 neurons, ReLU |
| Dropout | 0.3 |
| Output | 1 neuron, Sigmoid |

**Model Configuration:**

- Optimizer: Adam
- Loss Function: Binary Cross-Entropy
- Batch Size: 256
- Maximum Epochs: 10
- Early Stopping: Patience of 3
- Best Model Weights: Restored using EarlyStopping

The Embedding layer learns numerical word representations, while the LSTM captures sequential dependencies in customer reviews. Dropout layers help reduce overfitting.

### 4. Model Training

The model was trained using the training dataset and monitored using a separate validation dataset.

**Best Validation Accuracy: 93.04% (Epoch 3)**

Training was stopped after six epochs because validation loss stopped improving.

EarlyStopping restored the best-performing model weights.

### 5. Model Evaluation

The model evaluation pipeline includes:

- Test Accuracy and Loss
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Score
- Training and Validation Accuracy/Loss Curves

The final test metrics can be obtained by executing the evaluation cells in the notebook.

### 6. Sentiment Prediction

The trained model can classify new customer reviews using the same preprocessing and tokenization pipeline.

**Example Positive Review:**

> Absolutely love this product! The flavor is rich and authentic. Will definitely order again!

**Expected Sentiment:** Positive

**Example Negative Review:**

> Very disappointed. The item was stale and lacked flavor. Would not recommend.

**Expected Sentiment:** Negative

The classifier produces a probability score between 0 and 1. A threshold of 0.5 is used for binary sentiment classification.

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Amazon-Fine-Food-Sentiment-Analysis.git
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

The dataset is available on Kaggle:

[Amazon Fine Food Reviews Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

Alternatively, use KaggleHub:

```python
import kagglehub
import os
import pandas as pd

path = kagglehub.dataset_download("snap/amazon-fine-food-reviews")
df = pd.read_csv(os.path.join(path, "Reviews.csv"))
```

### 4. Run the Notebook

Open `Amazon_Fine_Food_Sentiment_Analysis.ipynb` in Google Colab or Jupyter Notebook and execute the cells sequentially.

For faster model training, a GPU-enabled environment is recommended.

## Key Learnings

- Applied NLP preprocessing techniques to real-world customer reviews.
- Implemented tokenization, vocabulary management, and sequence padding.
- Developed an LSTM-based deep learning model for binary sentiment classification.
- Used dropout and early stopping to reduce overfitting.
- Evaluated classification performance using multiple metrics.
- Learned the importance of data splitting and preventing information leakage.

## Future Improvements

- Compare LSTM performance with TF-IDF and Logistic Regression.
- Experiment with Bidirectional LSTM and GRU architectures.
- Explore pretrained word embeddings such as GloVe.
- Fine-tune transformer models such as BERT.
- Apply threshold tuning and class weighting to improve minority-class performance.
- Deploy the trained model using Streamlit or Flask.

## Author

**Satya Prakash Osho**

B.Tech — Metallurgical Engineering  
Indian Institute of Technology (BHU), Varanasi

**Areas of Interest:** Data Analytics, Machine Learning, Deep Learning, Natural Language Processing

**GitHub:** [osho-patel](https://github.com/osho-patel)

---

*This project demonstrates an end-to-end NLP and deep learning workflow for customer review sentiment classification.*
