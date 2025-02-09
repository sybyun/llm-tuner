import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_metric
from src.core.utils import load_config
from src.core.fine_tuning import apply_fine_tuning_method
from torch.utils.data import DataLoader
import torch

class TrainerModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.config = load_config("configs/model_config.yaml")

        # 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"]["name"])
        model = AutoModelForSequenceClassification.from_pretrained(self.config["model"]["name"], num_labels=2)

        # 파인튜닝 기법 적용
        self.model = apply_fine_tuning_method(model, self.config)

        # 평가 지표 로드
        self.metrics = {metric_name: load_metric(metric_name) for metric_name in self.config["metrics"]}

    def training_step(self, batch, batch_idx):
        inputs = self.tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        outputs = self.model(**inputs, labels=batch["label"])
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        outputs = self.model(**inputs, labels=batch["label"])
        preds = torch.argmax(outputs.logits, dim=-1)

        for metric_name, metric in self.metrics.items():
            metric.add_batch(predictions=preds, references=batch["label"])

        return {"preds": preds, "labels": batch["label"]}

    def validation_epoch_end(self, outputs):
        results = {metric_name: metric.compute() for metric_name, metric in self.metrics.items()}
        self.log_dict({f"val_{key}": value for key, value in results.items()})
        print(f"Validation Metrics: {results}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["training"]["learning_rate"])
        return optimizer

    def save_model(self, save_path="models/finetuned_model"):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"✅ 모델 저장 완료: {save_path}")
