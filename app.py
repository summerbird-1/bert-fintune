import string
import gradio as gr
import requests
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

model_dir = "my-bert-model"

config = AutoConfig.from_pretrained(model_dir, num_labels=3, finetuning_task="text-classification")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config)

def inference(input_text):
    inputs = tokenizer.batch_encode_plus(
                [input_text],
                max_length=512,
                pad_to_max_length=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    predicted_class_id = logits.argmax().item()
    output = model.config.id2label[predicted_class_id]
    return output

demo = gr.Interface(
    fn=inference,
    inputs=gr.Textbox(label="Input Text", scale=2, container=False),
    outputs=gr.Textbox(label="Output Label"),
    examples = [
        ["My last two weather pics from the storm on August 2nd. People packed up real fast after the temp dropped and winds picked up.", 1],
        ["Lying Clinton sinking! Donald Trump singing: Let's Make America Great Again!", 0],
        ],
    title="Tutorial: BERT-based Text Classificatioin",
    )

demo.launch(debug=True)