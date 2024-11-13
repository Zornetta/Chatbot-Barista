
from src.infrastructure.classifier import IntentClassifier
from src.infrastructure.nlp_processor import NLPProcessor
from src.interfaces.console import ConsoleUI
from src.application.service import ChatbotService
from src.infrastructure.repositories import JSONIntentRepository, JSONMenuRepository


def create_app():
    menuRepo = JSONMenuRepository("../../data/menu.json")
    intentRepo = JSONIntentRepository("../../data/intents.json")
    nLPProcessor = NLPProcessor(menuRepo.get_menu(), intentRepo.get_intents_dict())

    # Usar ruta por defecto para el modelo
    classifier = IntentClassifier(nLPProcessor)

    chatBotService = ChatbotService(
        menuRepo,
        intentRepo,
        nLPProcessor,
        classifier
    )
    return ConsoleUI(chatBotService)