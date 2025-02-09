import os
import yaml
import logging
import time
from datetime import datetime

# 로깅 설정
def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    return logging.getLogger()

# YAML 설정 파일 로드
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# 실행 시간 측정 데코레이터
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️ {func.__name__} 실행 시간: {end_time - start_time:.2f}초")
        return result
    return wrapper
