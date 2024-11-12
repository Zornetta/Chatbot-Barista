# src/infrastructure/persistence/file_model_storage.py
from typing import Any
from src.domain.training.interfaces import IModelStorage
import joblib
from pathlib import Path

class FileModelStorage(IModelStorage):
    def save_vectorizer(self, vectorizer: Any, path: str) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectorizer, path)
        return str(path)

    def load_vectorizer(self, path: str) -> Any:
        return joblib.load(path)