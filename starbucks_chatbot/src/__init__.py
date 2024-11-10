
from src.infrastructure.classifier import IntentClassifier
from src.infrastructure.nlp_processor import NLPProcessor
from src.interfaces.console import ConsoleUI
from src.application.service import ChatbotService
from src.infrastructure.repositories import JSONIntentRepository, JSONMenuRepository


def create_app():
    menuRepo = JSONMenuRepository("../../data/menu.json")
    intentRepo = JSONIntentRepository("../../data/intents.json")
    nLPProcessor = NLPProcessor(menuRepo.get_menu(), intentRepo.get_intents_dict())

    chatBotService = ChatbotService(
        menuRepo,
        intentRepo,
        nLPProcessor,
        IntentClassifier(nLPProcessor)
    )
    App = ConsoleUI(chatBotService)
    return App