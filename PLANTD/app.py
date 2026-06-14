import gradio as gr
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("plant_disease_model.h5")

# Class names (update this list according to your dataset)
class_names = ['Healthy', 'Diseased']

def predict(img):
    # Convert image to array
    img = img.resize((224, 224))  # resize to model input size
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = round(100 * np.max(predictions), 2)

    return {predicted_class: confidence}

# Create Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="🌱 Plant Disease Detection",
    description="Upload a plant leaf image to detect whether it's healthy or diseased."
)

if __name__ == "__main__":
    interface.launch()
