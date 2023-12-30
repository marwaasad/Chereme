import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import string
import operator
from googletrans import Translator

# Load the sign language detection models
asl_models = {
    
    'Indian Sign Language': tf.keras.models.load_model("C:\\Users\\Marwa Asad\\Desktop\\chereme\\models\\model_new.h5"),
    'American Sign Language': tf.keras.models.load_model("C:\\Users\\Marwa Asad\\Desktop\\chereme\\models\\model_epoch_8.h5"),
    #'Australian Sign Language': tf.keras.models.load_model("C:\\Users\\Marwa Asad\\Desktop\\chereme\\models\\model_new_auslan.h5")
}

# Load the dictionary of words and their corresponding spellings
with open("C:\\Users\\Marwa Asad\\Desktop\\chereme\\dictionary.txt") as f:
    dictionary = [line.strip() for line in f.readlines()]

def predict(test_image, language, text_lang):
    # Convert the image to a numpy array
    test_image = np.array(test_image)
    
    # Resize the input image to match the input shape of the model
    test_image = cv2.resize(test_image, (128, 128))
    
    # Get the appropriate sign language detection model based on the selected language
    model = asl_models[language]
    
    # Get the model prediction for the input image
    result = model.predict(test_image.reshape(1, 128, 128, 1))

    # Convert the prediction results to a dictionary of letter probabilities
    prediction = {}
    for i, letter in enumerate(string.ascii_uppercase):
        prediction[letter] = result[0][i+1]

    # Sort the letter probabilities in descending order
    sorted_prediction = sorted(prediction.items(), key=lambda x: x[1], reverse=True)

    # Get the predicted letter and corresponding word from the dictionary
    predicted_letter = sorted_prediction[0][0].lower()
    return predicted_letter

    predicted_word = dictionary.get(predicted_letter, 'blank')

    # Translate the predicted word to the desired text language
    translator = Translator(service_urls=['translate.googleapis.com'])
    translation = translator.translate(predicted_word, dest=text_lang)
    
    return translation.text


# Define the Streamlit app
def app():
    st.title('Sign Language to Text Conversion')
    
    # Add dropdowns to select the sign language model and text language
    language_options = ['American Sign Language', 'Indian Sign Language', 'Australian Sign Language']
    sign_language = st.selectbox('Select Sign Language:', options=language_options)
    
    text_language_options = ['English', 'Hindi', 'Spanish']
    text_language = st.selectbox('Select Text Language:', options=text_language_options)
    
    
    img_file = st.camera_input("Take a picture")
    if img_file is not None:
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        r = predict(cv2_img, sign_language, text_language)
        st.write(f'Recognized Text: {r}')

        #gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        #blur = cv2.GaussianBlur(gray, (5, 5), 2)
        #th3 = cv2.adaptiveThreshold(blur, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        #ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
        #recognized_text = predict(res, sign_language, text_language)
    
        # Display the recognized text on the screen
        #st.write(f'Recognized Text: {recognized_text}')
                    

# Run the Streamlit app
if __name__ == '__main__':
    app()
