````markdown
# Urdu Tweet Sentiment Analysis

## Introduction
This project is designed to perform sentiment analysis on Urdu tweets using a fine-tuned transformer-based deep learning model. The model classifies tweets into three categories:
- **Positive**
- **Neutral**
- **Negative**

The implementation leverages libraries such as [Hugging Face Transformers](https://huggingface.co/), [PyTorch](https://pytorch.org/), and [Gradio](https://gradio.app/) to build, train, and deploy the model through an interactive web interface.

---

## Project Overview
This repository includes:
- **Data Preprocessing & Balancing:**  
  Steps to clean the raw dataset, map raw sentiment labels into simplified categories, and create a balanced dataset.
  
- **Model Training:**  
  Code to tokenize data, set up training parameters, and fine-tune a pre-trained transformer (e.g., XLM-Roberta) for sentiment classification.
  
- **Evaluation:**  
  Generation of evaluation metrics (accuracy, precision, recall, F1 score, confusion matrix) and detailed classification reports.
  
- **Web Interface:**  
  A Gradio-based web app for real-time sentiment prediction on new Urdu tweets.
  
- **Model Saving & Inference:**  
  Scripts to save the trained model and tokenizer, and to perform predictions on new text inputs.

---

## Technologies Used
- **Python**: Main programming language
- **Transformers**: For pre-trained transformer models
- **PyTorch**: Deep learning framework for model training and inference
- **Pandas & NumPy**: Data manipulation and numerical operations
- **Scikit-learn**: Evaluation metrics and dataset splitting
- **Gradio**: User-friendly web interface for model inference
- **openpyxl**: Reading and writing Excel files (.xlsx)

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/urdu-tweet-sentiment-analysis.git
cd urdu-tweet-sentiment-analysis
```

### 2. Install Dependencies
Ensure you have Python 3.7 or later installed. Install required packages using:
```bash
pip install -r requirements.txt
```

### 3. Google Colab / Local Setup
- If using **Google Colab**, mount your Google Drive and install dependencies via pip commands in your notebook.
- For local development, ensure that all dependencies in `requirements.txt` are installed.

### 4. Pretrained Model and Tokenizer
The model is hosted on Hugging Face. You can load it in your code using:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = "abdullah123456/Sentiment_Analysis_Urdu_NLP"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

---

## Data Preparation
The raw dataset is stored in an Excel file containing Urdu tweets and their sentiment labels. The preprocessing steps include:

1. **Cleaning Text:**  
   Removing extra spaces and unwanted characters.
   ```python
   def clean_text(text):
       return " ".join(text.split())
   ```

2. **Mapping Raw Sentiment Labels:**  
   Converting detailed sentiment labels into three simplified categories (positive, neutral, negative) and mapping them to numeric IDs.
   ```python
   unique_labels = df['mapped_category'].dropna().unique().tolist()
   label2id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
   id2label = {idx: label for label, idx in label2id.items()}
   print("Mapped Label mapping:", label2id)
   ```

3. **Balancing the Dataset:**  
   Sampling an equal number of examples for each sentiment class.
   ```python
   subset_per_class = target_size // num_classes
   
   # Splitting dataset by classes
   df_neg = df[df['mapped_category'] == 'negative']
   df_neu = df[df['mapped_category'] == 'neutral']
   df_pos = df[df['mapped_category'] == 'positive']
   
   # Sampling each class
   df_neg_subset = df_neg.sample(n=subset_per_class, random_state=42)
   df_neu_subset = df_neu.sample(n=subset_per_class, random_state=42)
   df_pos_subset = df_pos.sample(n=subset_per_class, random_state=42)
   
   # Combining & shuffling
   df_subset_balanced = pd.concat([df_neg_subset, df_neu_subset, df_pos_subset], ignore_index=True)
   df_subset_balanced = df_subset_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
   ```

4. **Conversion to Hugging Face Dataset:**  
   This enables efficient tokenization, batching, and integration with the Transformers training pipeline.

5. **Splitting the Dataset:**  
   Dividing the dataset into training, validation, and test splits to ensure proper model evaluation and to prevent overfitting.

---

## Model Training
The model is fine-tuned using the following training parameters:

```python
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",  # Disables external logging (e.g., WandB)
    bf16=True         # Enables mixed precision training for speed and efficiency
)
```

### Key Concepts:
- **Learning Rate (2e-5):**  
  Sets the step size for weight updates during training. A small value is chosen to carefully fine-tune the pre-trained model without significant deviation.
  
- **Batch Size:**  
  Determines the number of samples processed in one training step. A batch size of 16 is used to balance memory usage and computational efficiency.
  
- **Evaluation Strategy:**  
  The model is evaluated at the end of each epoch to monitor performance and determine the best model based on the F1 score.
  
- **Mixed Precision Training:**  
  Uses both 16-bit and 32-bit floating point operations to speed up training and reduce memory usage without compromising model performance.

---

## Evaluation & Inference

### Evaluation:
- The model's performance is evaluated using metrics like accuracy, precision, recall, F1 score, and confusion matrix.
- A detailed classification report is generated with scikit-learn to assess performance on the test set.

### Inference:
- A prediction function cleans, tokenizes, and passes new Urdu tweets through the model to output a sentiment label.
- An interactive web interface is built using Gradio for real-time testing:
  ```python
  import gradio as gr

  iface = gr.Interface(
      fn=predict_sentiment,
      inputs=gr.Textbox(lines=4, placeholder="Enter an Urdu tweet here...", label="Urdu Tweet"),
      outputs=gr.Textbox(label="Predicted Sentiment"),
      title="Urdu Tweet Sentiment Analysis - 1",
      description="Enter your tweet below to see its sentiment prediction (Positive, Neutral, or Negative).",
      examples=[
          ["السلام علیکم! آج کا دن بہت خوبصورت ہے۔"],
          ["میں بہت غمگین ہوں، دل بہت دکھ رہا ہے۔"],
          ["آپ کا کام بہت اچھا ہے!"]
      ]
  )

  if __name__ == "__main__":
      iface.launch()
  ```

---

## File Structure
```
urdu-tweet-sentiment-analysis/
│
├── data/
│   └── Urdu Tweets Dataset.xlsx       # Raw dataset (Excel file)
│
├── models/
│   └── Trained Model/                  # Directory containing the saved model and tokenizer
│
├── results/                            # Training outputs and checkpoints
│
├── src/
│   ├── preprocess.py                   # Data preprocessing scripts
│   ├── train.py                        # Model training script
│   ├── inference.py                    # Inference and evaluation script
│   └── web_app.py                      # Gradio web interface code
│
├── README.md                           # Project documentation (this file)
├── requirements.txt                    # List of Python dependencies
└── app.py                              # Entry point to launch the Gradio interface
```

---

## Additional Concepts

### GPU vs. CPU
- **CPU:**  
  A general-purpose processor optimized for sequential and complex logic tasks.
- **GPU:**  
  A specialized processor with many cores designed for parallel processing, ideal for large-scale numerical computations in deep learning.

### WandB (Weights & Biases)
- A tool for tracking experiments, visualizing training progress, and managing hyperparameters. Although disabled in this project (`report_to="none"`), it is a valuable resource for monitoring and comparing model runs.

---

## Future Improvements
- **Expand Dataset:** Increase the dataset size for better model generalization.
- **Model Enhancement:** Experiment with different transformer models and architectures.
- **Deployment:** Explore deploying the model as an API for integration with other applications.
- **User Interface:** Enhance the web interface with additional features and visualizations.

---

## Contributing
Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request with your improvements.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments
- [Hugging Face](https://huggingface.co/) for providing state-of-the-art NLP models.
- [PyTorch](https://pytorch.org/) for the deep learning framework.
- [Gradio](https://gradio.app/) for making it easy to create interactive ML demos.
