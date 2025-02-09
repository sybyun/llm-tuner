import click
import yaml
from src.core.model_manager import ModelManager
from src.core.trainer import TrainerModule
import pytorch_lightning as pl

@click.command()
@click.option("--config", default="configs/model_config.yaml", help="Path to config file")
def train(config):
    with open(config, "r") as f:
        config_data = yaml.safe_load(f)
    
    model_manager = ModelManager(config_data)
    model = model_manager.model

    trainer = pl.Trainer(max_epochs=config_data["training"]["epochs"])
    trainer.fit(model)

if __name__ == "__main__":
    train()
