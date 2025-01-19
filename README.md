# Next-Word-Prediction-Using-LSTM-with-stramlit-deployment

This repository contains a next-word prediction model built using Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN). The project leverages TensorFlow/Keras for model development and Streamlit for creating a user-friendly web application interface for easy interaction with the model.

## Features

- Predict the next word in a sequence of text based on trained data.
- User-friendly interface deployed via Streamlit.
- Easily accessible and interactive platform for exploring natural language prediction.


## Demo

Access the deployed app here: [Next Word Prediction App](https://next-word-prediction-using-lstm-with-stramlit-deployment-g5a5c.streamlit.app/)

## **Below is a demonstration of the app's interface:**
![image](https://github.com/user-attachments/assets/87f4a94d-80d1-47d5-b108-6e98c4a61481)

## Repository Structure
- app.py: Main Python script containing the Streamlit application.
- next_word_lstm.h5: Pre-trained LSTM model saved in HDF5 format.
- tokenizer.pkl: Pickled tokenizer used for text pre-processing and sequence generation.
- requirements.txt: Python dependencies required for the project.
- README.md: Instructions and information about the project.
- lstm.ipynb: Jupyter Notebook for training the LSTM model.
- hemlet.txt: Sample text data used during the training phase.

## How to Use

1. Visit the App: Open the Streamlit app using this link: [Next Word Prediction App](https://next-word-prediction-using-lstm-with-stramlit-deployment-g5a5c.streamlit.app/)

2. Enter Text
   - Enter a sequence of words in the input text box.
   - For example: "The quick brown fox"
3. Predict Next Word
   - Click the "Predict Next Word" button.
   - The app will display the predicted next word based on the LSTM model.
  
## Local Setup Instructions

If you want to run this application locally, follow these steps:

1. Clone the Repository
   ``` bash
   git clone https://github.com/aljebraschool/Next-Word-Prediction-Using-LSTM-with-stramlit-deployment.git
   cd Next-Word-Prediction-Using-LSTM-with-stramlit-deployment
   ```
2. Install Dependencies
  Make sure you have Python installed. Then, install the required dependencies:
``` bash
pip install -r requirements.txt

```
3. Run the Application
  Start the Streamlit app:
``` bash
streamlit run app.py
```
The app will open in your browser, and you can interact with it locally.

## Model Description

The LSTM model is trained to predict the next word in a given sequence of text. It uses the following key components:

- Tokenizer: Maps words to integer indices.
- LSTM Layers: Captures sequential dependencies in the text.
- Dataset: The model was trained on the hemlet.txt dataset to learn word patterns and predict the next word in a sentence.

## Dependencies

- Python 3.7+
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pickle

For a complete list of dependencies, see the requirements.txt file.

### License

This project is licensed under the MIT License.

### Contributing

Contributions are welcome! If you'd like to improve the project or fix any issues, please feel free to submit a pull request.

### Author

This project was created by [Algebra School](aljebraschool.github.io).


  

