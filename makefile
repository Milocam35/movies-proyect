# Definir el nombre del entorno virtual
VENV_DIR = ML-Movies

# Comando para activar el entorno virtual en Windows
ACTIVATE = $(VENV_DIR)\Scripts\activate

# Objetivos
.PHONY: create_venv activate install run clean

.PHONY: deactivate

# Crear el entorno virtual si no existe
create_venv:
	python -m venv $(VENV_DIR)  # Esta línea debe tener un tabulador antes de 'python'

# Activar el entorno virtual
activate:
	cmd /c "$(ACTIVATE)"

deactivate:
	.\ML-Movies\Scripts\deactivate

# Instalar dependencias desde requirements.txt
install: activate
	pip install -r requirements.txt

# Ejecutar la aplicación principal
run: activate
	python training/src/main.py

# Limpiar el entorno virtual
clean:
	rm -rf $(VENV_DIR)
