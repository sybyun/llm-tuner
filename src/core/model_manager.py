from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class ModelManager:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        model_name = self.config["model"]["name"]
        model_type = self.config["model"]["type"]
        
        if model_type == "huggingface":
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            raise ValueError("Only 'huggingface' models are supported for now.")

        return model.to(self.device), tokenizer
