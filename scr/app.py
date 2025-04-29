import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model from a local path
model_path = "model\filadentification.h5"'
model = tf.keras.models.load_model(model_path)

# Define your class names
class_names = ["not-fila", "fila"]  # Adjusted order to match logic: 0 -> not-fila, 1 -> fila

# Prediction function
def predict(image):
    # Resize and normalize image
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict
    prediction = model.predict(image)

    # Handle sigmoid or softmax output
    if prediction.shape[-1] == 1:
        prob = prediction[0][0]
        predicted_class = class_names[int(prob > 0.5)]
        confidence = prob if prob > 0.5 else 1 - prob
    else:
        prob = tf.nn.softmax(prediction[0]).numpy()
        predicted_class = class_names[np.argmax(prob)]
        confidence = np.max(prob)

    confidence_percentage = confidence * 100
    result_text = f"Prediction: **{predicted_class}**\nConfidence: **{confidence_percentage:.2f}%**"
    return result_text

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Markdown(),
    title="Fila Yoruba Detector",
    description="""
        <div style='text-align: center'>
            <img src="https://github.com/DolapoSalim/my-phd-website-2/blob/adjusted/assets/images/favicons.jpg" width="100"/>
            <p>Upload an image. This model will predict whether the image contains the traditional Yoruba people's <strong>fila</strong> or not.</p>
        </div>
    """,
    theme="default"
)

# Run the app
interface.launch(share=True)