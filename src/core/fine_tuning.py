from peft import get_peft_model, LoraConfig, AdapterConfig, PrefixTuningConfig, BitFitConfig, TaskType
from transformers import AutoModelForSequenceClassification
import torch

def apply_fine_tuning_method(model, config):
    """
    주어진 모델에 대해 설정된 파인튜닝 방법을 적용합니다.
    Args:
        model: Hugging Face 모델
        config: 설정 파일에서 불러온 파인튜닝 관련 설정
    Returns:
        model: 파인튜닝이 적용된 모델
    """
    if config["model"]["fine_tuning_method"] == "LoRA" and config["lora"]["apply_lora"]:
        lora_config = LoraConfig(
            task_type=TaskType.SEQUENCE_CLASSIFICATION,
            r=config["lora"]["r"],
            alpha=config["lora"]["alpha"],
            dropout=config["lora"]["dropout"]
        )
        model = get_peft_model(model, lora_config)
        print("✅ LoRA 파인튜닝 적용됨")

    elif config["model"]["fine_tuning_method"] == "Adapter" and config["adapter"]["apply_adapter"]:
        adapter_config = AdapterConfig(
            hidden_size=config["adapter"]["adapter_size"]
        )
        model = get_peft_model(model, adapter_config)
        print("✅ Adapter 파인튜닝 적용됨")

    elif config["model"]["fine_tuning_method"] == "PrefixTuning" and config["prefix_tuning"]["apply_prefix_tuning"]:
        prefix_tuning_config = PrefixTuningConfig(
            prefix_length=config["prefix_tuning"]["prefix_length"]
        )
        model = get_peft_model(model, prefix_tuning_config)
        print("✅ Prefix Tuning 파인튜닝 적용됨")

    elif config["model"]["fine_tuning_method"] == "BitFit" and config["bitfit"]["apply_bitfit"]:
        bitfit_config = BitFitConfig()
        model = get_peft_model(model, bitfit_config)
        print("✅ BitFit 파인튜닝 적용됨")

    else:
        print("❗️파인튜닝 방법이 설정되지 않았거나 올바르지 않습니다.")

    return model
