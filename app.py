import streamlit as st
from PIL import Image
from keras.models import load_model
from keras.applications.vgg19 import preprocess_input
import numpy as np

# Load the pre-trained model
MODEL_PATH = r'C:\Users\user\Desktop\python\SUBHAM2208\project2\model_vgg19.h5'
model = load_model(MODEL_PATH)

# Define the Streamlit app
def main():
    st.title("Malaria-Infected Cell Classification")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        
        # Preprocess the image
        img = image.resize((224, 224))  # Resize image to match model's expected sizing
        img_array = np.array(img)  # Convert PIL image to numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)  # Preprocess image

        # Make prediction
        preds = model.predict(img_array)
        pred_class = np.argmax(preds, axis=1)
        
        if pred_class == 0:
            st.write("The cell is Infected with Malaria.")
        else:
            st.write("The cell is not Infected with Malaria.")

if __name__ == '__main__':
    main()

