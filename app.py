import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load model
model = load_model('next_word_LSTM.h5', compile=False)

# Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to generate next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):] #Ensuring the sequence length matches max_sequence_len
    
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
        
    return None 


# Streamlit app
st.title("Next Word Prediction Using the LSTM Model")

# User input
user_input = st.text_input("Enter text:","to be not to be")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1
    predicted_word = predict_next_word(model, tokenizer, user_input, max_sequence_len)
    st.write("Predicted next word:", predicted_word)
