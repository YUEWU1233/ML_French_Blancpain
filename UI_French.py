import streamlit as st
from transformers import CamembertTokenizer, CamembertModel
import torch
import joblib
import random
from PIL import Image
import pandas as pd
from lightgbm import LGBMClassifier
import random
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Dict to map numerical labels to their categorical equivalents
label_map = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}

# Set a question pool
questions = [
    "Please briefly introduce yourself.",
    "Describe your family.",
    "What are your hobbies?",
    "What is your favorite movie and why?",
    "What are your career plans for the future?",
    "Describe a memorable travel experience.",
    "What is your favorite book and why?",
    "What is your opinion on the current education system?",
    "Share your thoughts on technological development.",
    "Describe a new skill you would like to learn."
]

#use Cambert model to extract feature
def bert_feature(data, **kwargs):
    #load tokenizer and model
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    model = CamembertModel.from_pretrained('camembert-base', num_labels=6)
    # Tokenize and encode input texts
    inputs_ids = [tokenizer.encode(text, add_special_tokens=True, **kwargs) for text in data]

    # Extract BERT features for each input
    features = []
    with torch.no_grad():
        for inputs_id in inputs_ids:
            # Convert input ID to tensor
            inputs_tensor = torch.tensor([inputs_id])

            # Pass the tensor through the model
            output = model(inputs_tensor)
            # Extract the embeddings for the [CLS] token (index 0)
            cls_embedding = output.last_hidden_state[:, 0, :].numpy()

            # Add feature to list of all features
            features.append(cls_embedding)

    # Concatenate features from all inputs
    feature_data = np.concatenate(features, axis=0)

    # Clean up to free memory
    torch.cuda.empty_cache()

    return feature_data

# Load the tokenizer
#tokenizer = CamembertTokenizer.from_pretrained('camembert-base')

# Load the custom trained model and scaler
model_path = r"D:/Lausanne_Life/MA2/ML/French/results/三个数据集/SVC().pkl"
model = joblib.load(model_path)
scaler = joblib.load("D:/Lausanne_Life/MA2/ML/French/results/两个数据集/scaler.pkl")
def preprocess(text):
    """preprocess the text"""
    #  feature extraction and scale
    features = bert_feature([text])
    features_scaled = scaler.transform(features)
    
    return features_scaled

def evaluate_difficulty(text):
    """Evaluate the difficulty of a French text using the custom trained model."""
    features = preprocess(text)
    predictions = model.predict(features)
    predicted_class_id = predictions.argmax()
    print("Predictions:", predictions)  # 检查模型的预测结果
    print("Predicted class ID:", predicted_class_id)  # 检查最大预测值的索引
    difficulty_label = label_map[predicted_class_id]
    return difficulty_label

# Set the title of the app
st.title("French Level Testing")

# Randomly select a question from the question bank
selected_question = random.choice(questions)
st.write("Please write something in response to the following question:")
st.write(selected_question)

# Create a text input area for user input
user_input = st.text_area("Please write down here", height=300)

# Create a button to trigger difficulty evaluation
if st.button("Done"):
    if user_input:  # Check if there is any input
        difficulty_label = evaluate_difficulty(user_input)
        st.write(f"Your level is: {difficulty_label}")
    else:
        st.write("Please write something!")
