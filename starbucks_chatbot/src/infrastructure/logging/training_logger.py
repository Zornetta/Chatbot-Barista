# src/infrastructure/logging/training_logger.py
from src.domain.training.interfaces import ITrainingLogger
import logging

class ConsoleTrainingLogger(ITrainingLogger):
    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def error(self, message: str) -> None:
        self.logger.error(message)