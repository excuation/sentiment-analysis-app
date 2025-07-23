import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# Parameters
vocab_size = 10000
maxlen = 200
embedding_dim = 32
gru_units = 32

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# --- 1. Load and Preprocess Data ---
print("Loading and preprocessing data...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Padding sequences to the same length
x_train = pad_sequences(x_train, maxlen=maxlen, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=maxlen, padding='post', truncating='post')

# --- 2. Build a More Robust GRU Model ---
print("Building the model...")
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=maxlen),
    GRU(gru_units, return_sequences=True),  # Return sequences for stacking
    Dropout(0.5),  # Add dropout for regularization
    GRU(gru_units),
    Dropout(0.5),  # Add dropout for regularization
    Dense(1, activation='sigmoid') # Sigmoid for binary classification
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# --- 3. Set Up Callbacks ---
# Save only the best model based on validation accuracy
checkpoint = ModelCheckpoint(
    "models/best_sentiment_gru.h5",
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Stop training if validation loss doesn't improve for 3 epochs
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1
)

# --- 4. Train the Model ---
print("Training the model...")
history = model.fit(
    x_train, y_train,
    epochs=10,  # Can use more epochs because of EarlyStopping
    batch_size=128,
    validation_split=0.2,
    callbacks=[checkpoint, early_stopping] # Use the callbacks
)

# --- 5. Evaluate the Best Model ---
print("\nEvaluating the best model on the test set...")
# Load the best model saved by ModelCheckpoint
model.load_weights("models/best_sentiment_gru.h5")
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {accuracy:.4f}")
print("✅ Model training complete and best model saved to 'models/best_sentiment_gru.h5'.")git remote add origin https://github.com/excuation/sentiment-analysis-app.git


