import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import string
import numpy as np

# --- Configuration ---
VOCAB_SIZE = 10000
MAXLEN = 200
MODEL_PATH = "models/best_sentiment_gru.h5" # Use the best model

# --- Caching Functions for Performance ---

# Cache the model loading, so it only runs once
@st.cache_resource
def load_model():
    """Loads the pre-trained Keras model."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error(f"Please make sure you have trained the model using 'train_improved.py' and the '{MODEL_PATH}' file exists.")
        return None

# Cache the word index, so it's only loaded once
@st.cache_data
def load_word_index():
    """Loads the IMDB word index."""
    word_to_index = imdb.get_word_index()
    # The first indices are reserved, so we offset by 3
    word_to_index = {k:(v+3) for k,v in word_to_index.items()}
    word_to_index["<PAD>"] = 0
    word_to_index["<START>"] = 1
    word_to_index["<UNK>"] = 2  # Unknown word
    word_to_index["<UNUSED>"] = 3
    return word_to_index

# --- Preprocessing ---

def encode_review(text, word_to_index):
    """
    Encodes a user's text review into a sequence of integers.
    This version correctly handles punctuation.
    """
    # 1. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 2. Convert to lowercase and split into words
    words = text.lower().split()
    # 3. Encode words to integers, using <UNK> for unknown words
    encoded = [word_to_index.get(word, 2) for word in words]
    # 4. Prepend the <START> token
    encoded = [1] + encoded
    # 5. Pad the sequence
    padded = pad_sequences([encoded], maxlen=MAXLEN, padding='post', truncating='post')
    return padded

# --- Streamlit UI ---

st.set_page_config(page_title="IMDB Sentiment Analysis", layout="centered")
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.markdown("Enter a movie review below. The model will predict whether the sentiment is positive or negative.")

# Load resources
model = load_model()
word_index = load_word_index()

# Only proceed if the model loaded successfully
if model:
    # Example buttons
    st.markdown("##### Try an example:")
    col1, col2 = st.columns(2)
    if col1.button("Positive Review Example", use_container_width=True):
        example_text = "I loved this movie! The acting was incredible and the plot was so engaging. I would recommend this to everyone."
    elif col2.button("Negative Review Example", use_container_width=True):
        example_text = "This was a complete waste of time. The story was predictable and the characters were boring. I would not watch this again."
    else:
        example_text = ""

    # User input
    user_input = st.text_area("Your review here:", value=example_text, height=150)

    if st.button("Analyze Sentiment", use_container_width=True, type="primary"):
        if user_input:
            # 1. Preprocess the input
            input_seq = encode_review(user_input, word_index)
            # 2. Make a prediction
            pred = model.predict(input_seq)[0][0]
            sentiment = "Positive 😊" if pred >= 0.5 else "Negative 😞"

            # 3. Display the result
            st.write("---")
            st.write("**Prediction:**")
            if sentiment == "Positive 😊":
                st.success(f"**Sentiment: {sentiment}**")
            else:
                st.error(f"**Sentiment: {sentiment}**")

            st.write("**Confidence Score:**")
            st.progress(float(pred))
            st.write(f"The model is {pred:.2%} confident that the review is positive.")
        else:
            st.warning("Please enter a review to analyze.")

