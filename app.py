import joblib
import gradio as gr

# Load pre-trained pipeline
loaded_pipeline = joblib.load('spam_detection_pipeline.joblib')

# Define input and output interfaces
input_interface = gr.inputs.Textbox(label="Email Message")
output_interface = gr.outputs.Textbox(label="Prediction")

# Define prediction function
def predict_spam(message):
    prediction = loaded_pipeline.predict([message])
    return "Spam" if prediction == 1 else "Not Spam"

# Create Gradio interface
app = gr.Interface(fn=predict_spam, inputs=input_interface, outputs=output_interface, title="Email Spam Classifier")

# Launch the interface
app.launch()
