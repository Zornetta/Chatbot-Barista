#!/usr/bin/env python3

from src.application.service import ChatbotService
from src.infrastructure.classifier import IntentClassifier
from src.infrastructure.nlp_processor import NLPProcessor
from src.infrastructure.repositories import JSONIntentRepository, JSONMenuRepository


def test_chatbot_interaction():
    """Ejemplo de interacción con el chatbot"""
    # 1. Configurar componentes
    menu_repo = JSONMenuRepository("../../data/menu.json")
    intent_repo = JSONIntentRepository("../../data/intents.json")

    # 2. Inicializar NLP y Classifier
    nlp_processor = NLPProcessor(
        menu_repo.get_menu(),
        intent_repo.get_intents_dict(),
        "models/vectorizer.joblib"
    )

    intent_classifier = IntentClassifier(
        nlp_processor,
        "models/classifier/latest/classifier.joblib"
    )

    # 3. Crear servicio
    chatbot = ChatbotService(
        menu_repo=menu_repo,
        intent_repo=intent_repo,
        nlp_processor=nlp_processor,
        intent_classifier=intent_classifier
    )

    # 4. Simular interacción
    conversations = [
        # Consulta de menú
        ("¿Qué bebidas tienen?", "consultar_menu"),

        # Orden simple
        ("Quiero un latte grande", "ordenar_bebida"),

        # Personalización
        ("con leche de almendra", "ordenar_bebida"),

        # Consulta de precio
        ("¿Cuánto cuesta el cappuccino?", "preguntar_precio"),

        # Agregar a la orden
        ("También quiero un muffin de arándanos", "ordenar_comida"),

        # Confirmar orden
        ("Listo, eso sería todo", "confirmar_orden"),

        # Confirmación final
        ("Sí, confirmo", "confirmar_orden")
    ]

    print("=== Iniciando demo del chatbot ===\n")

    for user_input, expected_intent in conversations:
        print(f"\nUsuario: {user_input}")

        # Procesar mensaje
        response = chatbot.process_message(user_input)

        # Mostrar respuesta
        print(f"Starbucks: {response.text}")

        if response.suggested_actions:
            print("\nAcciones sugeridas:")
            for action in response.suggested_actions:
                print(f"- {action}")

        if response.order:
            print("\nOrden actual:")
            print(f"Items: {len(response.order.items)}")
            print(f"Total: ${response.order.total:.2f}")

        print("\n" + "="*50)

if __name__ == "__main__":
    test_chatbot_interaction()