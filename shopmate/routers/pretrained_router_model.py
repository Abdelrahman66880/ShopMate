# pretrained router model file
from .enums.ClassesEnums import ClassesEnums
from .enums.LLMsModelsEnums import LLMsModelsEnums
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn

class SemanticRouter:
    def __init__(self):
        model_name = LLMsModelsEnums.MODEL_NAME.value
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        product_prob = probs[0][1].item()
        return ClassesEnums.PRODUCT_ROUTE_NAME.value if product_prob > 0.5 else ClassesEnums.CHITCHAT_ROUTE_NAME.value, product_prob

    def guide(self, query: str) -> str:
        result, confidence = self.predict(query)
        return result

