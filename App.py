import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Page configuration
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="𝕏",
    layout="centered"
)

# Model configuration
MODEL_PATH = "./model"

LABELS = {
    0: "Irrelevant",
    1: "Negative",
    2: "Neutral",
    3: "Positive"
}

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Application UI
st.title("𝕏 Twitter Sentiment Analysis")
st.write(
    "Analyze the sentiment of tweets using a fine-tuned BERT model."
)

tweet = st.text_area(
    "Enter your tweet:",
    placeholder="Type your tweet here...",
    height=120
)

if st.button("Analyze Sentiment", use_container_width=True):

    if not tweet.strip():
        st.warning("Please enter a tweet.")

    else:
        inputs = tokenizer(
            tweet,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=1)

        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item() * 100

        sentiment = LABELS[prediction]

        st.subheader("Prediction Result")
        st.success(f"Sentiment: {sentiment}")
        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}%"
        )

        st.write("Class probabilities:")

        for label_id, label in LABELS.items():
            score = probabilities[0][label_id].item()
            st.write(f"{label}: {score:.2%}")
            st.progress(score)
