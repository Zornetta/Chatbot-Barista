Repo para implementar el chat barista

starbucks_chatbot/
├── src/
│   ├── domain/         # Lógica de negocio
│   ├── infrastructure/ # Implementaciones concretas
│   ├── application/    # Casos de uso
│   └── interfaces/     # APIs y UI
├── data/              # Datos de entrenamiento
├── tests/
└── docs/              # Documentación y diagramas

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows:
venv\Scripts\activate
# Unix/MacOS:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt