#!/usr/bin/env python3

# src/interfaces/console.py
from src.application.service import ChatbotService
from src.application.models import Response

class ConsoleUI:
    def __init__(self, chatbot: ChatbotService):
        self.chatbot = chatbot

    def run(self):
        """Inicia la interfaz de consola"""
        print("¡Bienvenido a Starbucks! ¿En qué puedo ayudarte?")
        print("(Escribe 'salir' para terminar)")

        while True:
            # Obtener input del usuario
            user_input = input("\nTú: ").strip()

            if user_input.lower() == 'salir':
                print("\n¡Gracias por tu visita! ¡Hasta pronto!")
                break

            # Procesar mensaje y mostrar respuesta
            response = self.chatbot.process_message(user_input)
            self.display_response(response)

    def display_response(self, response: Response):
        """Muestra la respuesta del chatbot"""
        print("\nStarbucks:", response.text)

        if response.suggested_actions:
            print("\nAcciones sugeridas:")
            for i, action in enumerate(response.suggested_actions, 1):
                print(f"{i}. {action}")

        if response.order:
            print("\nTu orden actual:")
            for item in response.order.items:
                print(f"- {item.menu_item.name} ({item.size})")
            print(f"Total: ${response.order.total:.2f}")