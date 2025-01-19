import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
from tensorflow.keras.models import load_model

try:
    #load the trained model
    model = load_model('next_word_lstm.h5')
except Exception as e:
    st.error(f"filed to laod model: {e}")
    st.stop()

try:
    #load the tokenizer 
    with open("tokenizer.pkl", 'rb') as file:
        tokenizer = pickle.load(file)
except Exception as e:
    st.error(f"filed to load tokenizer: {e}")
    st.stop()


#Function to predict the next word
def predict_next_word(model, tokenizer, text, sentence_length):
    # each each sequence of text passed, obtain the mapped integer to each word
    #this returns list of list, so select the first index of the list using [0]
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    # check if the token list returned is longer than the length of sentence used to train the model
    # adjust the token list length to take one less than the size of the sentence length but from behind using (-)
    if len(token_list) >= sentence_length:
        token_list = token_list[-(sentence_length-1):]
    
    # if token list length is not longer than sentence length, then pad it
    #note : sentence_length -1 is taking one element less than the length of the sentece_length
    # that is if sentence_length is 5 then this will take 4
    padded_sequence = pad_sequences([token_list], maxlen=sentence_length-1, padding='pre')
    
    # Get prediction
    predicted = model.predict(padded_sequence, verbose=0)
    #get the list index with the highest prediction
    #use [0] because this returns list of list
    predicted_index = np.argmax(predicted[0])  # Changed to [0] since predict returns a batch
    
    # Convert index back to word
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return None

#streamlit title
st.title("Next Word Prediction Using LSTM")

user_input = st.text_input("Enter sequence of text")

try:
    
    sentence_length = model.input_shape[1] + 1
except Exception as e:
    st.error(f"mismatched sentence length: {e}")
    st.stop()

if user_input:
    
    if st.button("Predict Next Word"):
        next_word = predict_next_word(model, tokenizer, user_input, sentence_length)

        st.write(f"Input text: {user_input}")
        st.write(f"Predicted Next Word : {next_word} ")

else:
    st.write(f"Please enter input text")


# Test function
def test_model():
    print("Testing Next Word Prediction Model...\n")
    
    test_inputs = [
        "The quick brown",
        "Machine learning is",
        "Artificial intelligence and",
        "",
        "Deep learning models are"
    ]
    
    for i, input_text in enumerate(test_inputs, 1):
        print(f"Test Input #{i}:")
        print(f"Input Text: {input_text}")
        if input_text.strip():
            predicted_word = predict_next_word(model, tokenizer, input_text, sentence_length)
            if predicted_word:
                print(f"Predicted Next Word: {predicted_word}\n")
            else:
                print("No prediction available (possibly due to out-of-vocabulary words).\n")
        else:
            print("No input text provided. Please enter valid text.\n")

if __name__ == "__main__":
    test_model()