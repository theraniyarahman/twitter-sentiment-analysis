# Twitter Sentiment Analysis using Fine-Tuned BERT

## Project Overview

This project uses a fine-tuned BERT (Bidirectional Encoder Representations from Transformers) model to classify the sentiment of Twitter posts.

The model analyzes the text of a tweet and predicts one of four sentiment categories: Positive, Negative, Neutral, or Irrelevant. A Streamlit web application provides an interactive interface where users can enter a tweet and view its predicted sentiment along with the corresponding class probabilities.

The project demonstrates an end-to-end NLP workflow, covering data preprocessing, transformer fine-tuning, model evaluation, application development, and containerization with Docker.

## Objectives

- Build a text classification model using BERT.
- Preprocess and prepare Twitter sentiment data.
- Fine-tune a pretrained BERT model for four-class sentiment classification.
- Evaluate model performance using classification metrics.
- Develop an interactive application using Streamlit.
- Package the application using Docker for portability.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| NLP Model | BERT (`bert-base-uncased`) |
| Deep Learning | PyTorch |
| NLP Framework | Hugging Face Transformers |
| Data Processing | Pandas, NumPy |
| Model Evaluation | Scikit-learn |
| Web Application | Streamlit |
| Containerization | Docker |
| Version Control | Git, GitHub |
| Model Hosting (planned) | Hugging Face Hub |

## Dataset

**Dataset:** Twitter Entity Sentiment Analysis

**Source:** [Kaggle – Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)

The dataset contains tweets labelled with four sentiment categories:

- Positive
- Negative
- Neutral
- Irrelevant

The dataset includes tweet IDs, entities, sentiment labels, and tweet text.

For this implementation, the model uses the tweet text as its input and predicts the sentiment category.

## Project Workflow

1. **Data Collection:** Obtain the training and validation datasets from Kaggle.
2. **Data Preprocessing:** Handle missing values, duplicates, and inconsistent records.
3. **Label Encoding:** Convert sentiment categories into numerical labels.
4. **Tokenization:** Use the BERT tokenizer to convert tweet text into token IDs and attention masks.
5. **Model Fine-Tuning:** Fine-tune `bert-base-uncased` for four-class sentiment classification.
6. **Model Evaluation:** Evaluate predictions using accuracy, precision, recall, and F1-score.
7. **Application Development:** Build a Streamlit interface for interactive sentiment prediction.
8. **Containerization:** Package the application and its dependencies using Docker.

## Model Details

- **Base Model:** `bert-base-uncased`
- **Architecture:** BERT for sequence classification
- **Task:** Multi-class text classification
- **Number of Classes:** 4
- **Maximum Sequence Length:** 128 tokens

### Label Mapping

| Sentiment | Encoded Label |
|---|---:|
| Irrelevant | 0 |
| Negative | 1 |
| Neutral | 2 |
| Positive | 3 |

## Model Evaluation

The model was evaluated on 1,000 records from the provided validation dataset.

| Metric | Score |
|---|---:|
| Accuracy | 0.9630 |
| Macro F1-Score | 0.9627 |

These are the observed evaluation results for the selected validation sample. They should not be interpreted as results on an independently held-out test set unless the validation data was kept entirely separate from training and model selection.

## Streamlit Application

The application allows users to:

- Enter a tweet or short text.
- Generate a sentiment prediction.
- View the predicted sentiment category.
- View the model's probability scores across all four classes.

The application uses the fine-tuned BERT model and tokenizer for inference.

## Project Structure

```text
Twitter-Sentiment-Analysis/
│
├── App.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── README.md
│
├── notebooks/
│   └── Fine_Tuning_BERT.ipynb
│
└── model/                  # Local model files
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── training_args.bin
```

**Note:** The trained model directory is excluded from Git to avoid committing large model files. The model will be hosted separately on Hugging Face Hub for cloud deployment.

## Installation and Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/theraniyarahman/Twitter-Sentiment-Analysis.git
cd Twitter-Sentiment-Analysis
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the model

Download or obtain the trained model files and place them in the expected model directory, or configure the application to load the model from Hugging Face Hub.

### 5. Run the application

```bash
streamlit run App.py
```

Open the local application at:

```text
http://localhost:8501
```

## Docker Setup

### Build the Docker image

```bash
docker build -t twitter-sentiment-bert .
```

### Run the container

```bash
docker run -d -p 8501:8501 --name twitter-sentiment-app twitter-sentiment-app
```

Open:

```text
http://localhost:8501
```

The Docker image is also published on Docker Hub.

**Docker Hub:** theraniyarahman/twitter_sentiment_analysis

## Future Improvements

- Deploy the Streamlit application using Streamlit Community Cloud.
- Host the fine-tuned model on Hugging Face Hub.
- Evaluate the model on a separate, untouched test dataset.
- Perform error analysis on misclassified tweets.
- Explore confidence calibration and prediction reliability.
- Improve handling of informal language, slang, and ambiguous tweets.

## Learning Outcomes

Through this project, I explored:

- NLP preprocessing and text classification.
- Transformer architecture and BERT fine-tuning.
- Transfer learning using pretrained language models.
- Model evaluation using classification metrics.
- Building an interactive ML application with Streamlit.
- Containerizing a machine learning application with Docker.

## Author

**RANIYA RAHMAN T.R**

GitHub: [theraniyarahman](https://github.com/theraniyarahman)
