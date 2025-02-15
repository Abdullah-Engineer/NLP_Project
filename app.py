import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Defining the model path relative to the repository root.
model_path = "fine_tuned_model"

# Loading the model and tokenizer from the local directory.
# The parameter `local_files_only=True` ensures that the files are loaded from the repository.
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

# Defining a simple text cleaning function
def clean_text(text):
    return " ".join(text.split())

# Defining the prediction function that the web interface will use.
def predict_sentiment(tweet: str) -> str:

    # Cleaning the tweet
    tweet_clean = clean_text(tweet)

    # Tokenizing the tweet. 
    inputs = tokenizer(tweet_clean, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

    # Moving the input tensors to the same device as the model.
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    # Getting the predicted class index.
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    
    # Defining label mapping.
    label_mapping = {0: "negative", 1: "neutral", 2: "positive"}
    return label_mapping.get(predicted_class, "unknown")

# Creating the Gradio Interface.
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Enter an Urdu tweet here...", label="Urdu Tweet"),
    outputs=gr.Textbox(label="Predicted Sentiment"),
    title="Urdu Tweet Sentiment Analysis",
    description="This app uses a fine-tuned transformer model to predict the sentiment of Urdu tweets. "
                "Enter your tweet in the textbox below and click 'Submit' to see the prediction.",
    examples=[
        ["السلام علیکم! آج کا دن بہت خوبصورت ہے۔"],
        ["میں بہت غمگین ہوں، دل بہت دکھ رہا ہے۔"],
        ["آپ کا کام بہت اچھا ہے!"]
    ]
)

# Launching the interface.
if __name__ == "__main__":
    iface.launch()