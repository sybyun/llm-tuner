import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split

def split_dataset(data, test_size=0.2, val_size=0.1):
    """
    데이터셋을 train, validation, test로 분할합니다.
    
    Args:
        data (dict): {"text": [...], "label": [...]}
        test_size (float): 테스트 데이터 비율
        val_size (float): 검증 데이터 비율
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        data["text"], data["label"], test_size=test_size, random_state=42
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=val_size / test_size, random_state=42
    )

    return (
        Dataset.from_dict({"text": train_texts, "label": train_labels}),
        Dataset.from_dict({"text": val_texts, "label": val_labels}),
        Dataset.from_dict({"text": test_texts, "label": test_labels}),
    )

def collate_fn(batch, tokenizer):
    """
    배치를 처리하여 모델에 입력할 수 있도록 변환합니다.
    
    Args:
        batch (list): 데이터 샘플 리스트
        tokenizer (AutoTokenizer): Hugging Face 토크나이저 객체
    
    Returns:
        dict: 토큰화된 입력 데이터
    """
    texts = [item["text"] for item in batch]
    labels = [item["label"] for item in batch]
    
    tokenized_inputs = tokenizer(
        texts, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
    )
    
    tokenized_inputs["labels"] = torch.tensor(labels)
    return tokenized_inputs

def create_data_loader(dataset: Dataset, tokenizer_name: str, batch_size=16):
    """
    주어진 데이터셋을 PyTorch DataLoader로 변환합니다.
    
    Args:
        dataset (Dataset): Hugging Face Dataset 객체
        tokenizer_name (str): 사용할 토크나이저 이름 (예: "bert-base-uncased")
        batch_size (int, optional): 배치 크기 (기본값 16)
    
    Returns:
        DataLoader: PyTorch DataLoader 객체
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, tokenizer))

def load_and_preprocess_data(dataset_name: str, tokenizer_name: str, test_size=0.2, val_size=0.1, batch_size=16):
    """
    데이터셋을 로드하고 train/val/test로 분할한 후, DataLoader를 생성합니다.
    
    Args:
        dataset_name (str): Hugging Face 데이터셋 이름 (예: "imdb")
        tokenizer_name (str): 사용할 토크나이저 이름
        test_size (float, optional): 테스트 데이터 비율 (기본값 0.2)
        val_size (float, optional): 검증 데이터 비율 (기본값 0.1)
        batch_size (int, optional): 배치 크기 (기본값 16)
    
    Returns:
        dict: {"train": train_loader, "val": val_loader, "test": test_loader}
    """
    # 데이터셋 로드
    dataset = load_dataset(dataset_name)

    # Train/Val/Test 데이터 분할
    train_dataset, val_dataset, test_dataset = split_dataset(
        {"text": dataset["train"]["text"], "label": dataset["train"]["label"]},
        test_size=test_size,
        val_size=val_size
    )

    # DataLoader 생성
    train_loader = create_data_loader(train_dataset, tokenizer_name, batch_size)
    val_loader = create_data_loader(val_dataset, tokenizer_name, batch_size)
    test_loader = create_data_loader(test_dataset, tokenizer_name, batch_size)

    return {"train": train_loader, "val": val_loader, "test": test_loader}
