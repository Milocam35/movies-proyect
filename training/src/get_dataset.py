import os
import json
import requests

# Lista de URLs y nombres de archivos JSON para descargar
urls = {
    "metadata.json": "URL_DEL_METADATA_JSON",
    "ratings.json": "URL_DEL_RATINGS_JSON",
    "reviews.json": "URL_DEL_REVIEWS_JSON",
    "survey_answers.json": "URL_DEL_SURVEY_ANSWERS_JSON",
    "tag_count.json": "URL_DEL_TAG_COUNT_JSON",
    "tags.json": "URL_DEL_TAGS_JSON"
}

# Directorio donde se guardarán los datos raw
raw_data_dir = "raw"

# Crear el directorio si no existe
if not os.path.exists(raw_data_dir):
    os.makedirs(raw_data_dir)

# Descargar y guardar cada archivo JSON
for filename, url in urls.items():
    response = requests.get(url)
    file_path = os.path.join(raw_data_dir, filename)
    with open(file_path, 'wb') as f:
        f.write(response.content)
    print(f"Descargado {filename} y guardado en {file_path}")

# Cargar los archivos JSON en Python
data = {}
for filename in urls.keys():
    file_path = os.path.join(raw_data_dir, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        data[filename] = json.load(f)

# Ejemplo de cómo acceder a los datos
metadata = data['metadata.json']
ratings = data['ratings.json']
reviews = data['reviews.json']
survey_answers = data['survey_answers.json']
tag_count = data['tag_count.json']
tags = data['tags.json']

print("Datos cargados exitosamente.")