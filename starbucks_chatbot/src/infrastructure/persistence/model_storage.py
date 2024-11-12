# src/infrastructure/persistence/model_storage.py
from src.domain.training.interfaces import IModelStorage
from pathlib import Path
from datetime import datetime
import joblib
import shutil
import os

class FileModelStorage(IModelStorage):
    def __init__(self, base_path: str = "models/vectorizer"):
        self.base_path = Path(base_path)
        self.latest_path = self.base_path / "latest"
        self.archive_path = self.base_path / "archive"

        # Crear directorios si no existen
        self.latest_path.mkdir(parents=True, exist_ok=True)
        self.archive_path.mkdir(parents=True, exist_ok=True)

    def save_vectorizer(self, vectorizer: Any, path: str = None) -> str:
        """Guarda el vectorizador y mantiene una copia en latest"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if path is None:
            # Guardar en archive con timestamp
            path = self.archive_path / f"vectorizer_{timestamp}.joblib"
        else:
            path = Path(path)

        # Guardar modelo
        joblib.dump(vectorizer, path)

        # Actualizar latest
        latest_file = self.latest_path / "vectorizer.joblib"
        shutil.copy2(path, latest_file)

        return str(path)

    def load_vectorizer(self, path: str = None) -> Any:
        """Carga el vectorizador (por defecto el más reciente)"""
        if path is None:
            path = self.latest_path / "vectorizer.joblib"

        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encuentra el modelo en: {path}")

        return joblib.load(path)