Repo para implementar el chat barista

starbucks_chatbot/
├── src/
│   ├── domain/         # Lógica de negocio
│   ├── infrastructure/ # Implementaciones concretas
│   ├── application/    # Casos de uso
│   └── interfaces/     # APIs y UI
├── tests/
│   ├── domain/         # test que validan Lógica de negocio
│   ├── infrastructure/ # test que validan Implementaciones concretas
│   ├── application/    # test que validan Casos de uso
│   └── interfaces/     # test que validan APIs y UI
│   __init__.py
├── data/              # Datos de entrenamiento
├── models/              # Modelos entrenados
├── scripts/
│   └── train_models.py  # Script de entrenamiento
├── docs/              # Documentación y diagramas
├── analyze_solution.py
└── app.py


<!-- Si quiero analizar el proyecto -->
# Analizar proyecto
python analyze_solution.py --definition_path=analyze_solution_definition.json

<!-- Si quiero correr el proyecto -->
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows:
venv\Scripts\activate
# Unix/MacOS:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

<!-- Si quiero crear los modelos, en el folder scripts/*.py  -->
<!-- Primera vez  -->

# Crear entorno virtual
python -m venv venv

# 1. Activar entorno virtual (si no está activo)
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Unix/MacOS

# 2. Instalar el paquete
pip install -e .

# 3. Ejecutar entrenamiento
python scripts/train_models.py

<!-- Si quiero hacer un cambio, secuencia  -->

# 1. Desinstalar el paquete actual (por si acaso)
pip uninstall starbucks_chatbot

# 2. Reinstalar en modo desarrollo
pip install -e .

# 3. Verificar instalación
pip list | grep starbucks_chatbot

# 4. Ejecutar el script de nuevo
python scripts/train_models.py
python scripts/train_classifier.py