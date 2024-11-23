from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, XLNetTokenizer, RobertaTokenizer, BertTokenizer, T5Tokenizer

# Initialize FastAPI app
app = FastAPI()

# Set up Jinja2 templates for rendering HTML
templates = Jinja2Templates(directory="templates")

# Load the ensemble weights and logits (from the saved model file)
ensemble_model_path = "best_model.pt"
weights = None
logits = None

try:
    print("Loading model weights and logits...")
    model_data = torch.load(ensemble_model_path, map_location=torch.device("cpu"))
    weights = model_data['weights']
    logits = model_data['logits']

    print(f"Loaded weights: {weights}")
    print(f"Loaded logits for XLNet: {logits['xlnet'][:5]}")
    print(f"Loaded logits for RoBERTa: {logits['roberta'][:5]}")
    print(f"Loaded logits for BERT: {logits['bert'][:5]}")
    print(f"Loaded logits for T5: {logits['t5'][:5]}")

except Exception as e:
    print(f"Error loading model data: {e}")

# Initialize tokenizers for each model
tokenizer_xlnet = XLNetTokenizer.from_pretrained("xlnet-base-cased")
tokenizer_roberta = RobertaTokenizer.from_pretrained("roberta-base")
tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer_t5 = T5Tokenizer.from_pretrained("t5-small")

# Initialize models for each transformer
model_xlnet = AutoModelForSequenceClassification.from_pretrained("xlnet-base-cased")
model_roberta = AutoModelForSequenceClassification.from_pretrained("roberta-base")
model_bert = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
model_t5 = AutoModelForSequenceClassification.from_pretrained("t5-small")

# Set models to evaluation mode
model_xlnet.eval()
model_roberta.eval()
model_bert.eval()
model_t5.eval()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the input form."""
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})


@app.post("/", response_class=HTMLResponse)
async def predict_bias(request: Request, sentence: str = Form(...)):
    """Handle prediction for the submitted sentence."""
    try:
        print("\n=== New Prediction Request ===")
        print(f"Input Sentence: {sentence}")

        # Tokenize input for all models
        xlnet_input = tokenizer_xlnet(sentence, return_tensors='pt', max_length=128, padding=True, truncation=True)
        roberta_input = tokenizer_roberta(sentence, return_tensors='pt', max_length=128, padding=True, truncation=True)
        bert_input = tokenizer_bert(sentence, return_tensors='pt', max_length=128, padding=True, truncation=True)
        t5_input = tokenizer_t5(sentence, return_tensors='pt', max_length=128, padding=True, truncation=True)

        # Get logits from each model
        with torch.no_grad():
            xlnet_logits = model_xlnet(**xlnet_input).logits
            roberta_logits = model_roberta(**roberta_input).logits
            bert_logits = model_bert(**bert_input).logits
            t5_logits = model_t5(**t5_input).logits

        # Combine the logits from all models using the saved weights
        ensemble_logits = (
            weights[0] * xlnet_logits +
            weights[1] * roberta_logits +
            weights[2] * bert_logits +
            weights[3] * t5_logits
        )

        # Get the final prediction from ensemble logits
        ensemble_prediction = np.argmax(ensemble_logits.detach().numpy(), axis=-1)[0]

    
        if ensemble_prediction == 1:
            result = "Biased"
        else:
            result = "Not Biased"

        print(f"Prediction: {result}")

        return templates.TemplateResponse(
            "index.html", {"request": request, "prediction": result, "sentence": sentence}
        )

    except Exception as e:
        print(f"Error during prediction: {e}")
        return templates.TemplateResponse(
            "index.html", {"request": request, "prediction": f"Error: {str(e)}", "sentence": sentence}
        )


def test_model():
    """Quick utility to test the model."""
    sample_input = "The economy is booming."
    inputs = tokenizer_bert(sample_input, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model_bert(**inputs)
        print(f"Logits for '{sample_input}': {output.logits}")
