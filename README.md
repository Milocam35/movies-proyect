# Proyecto de Recomendación de Películas Basado en Contenido
Para este proyecto, se busca realizar un sistema de recomendación de películas basado en contenido, utilizando utilizando datos crudos de varias fuentes como MovieLens - IMDB - encuestas de usuarios, el sistema proporcionará recomendaciones personalizadas a los usuarios basándose en sus preferencias de contenido, utilizando un enfoque de MLOps para automatizar el desarrollo, despliegue y monitoreo de modelos de machine learning.

# Datos utilizados
*MovieLens Tag Genome Dataset 2021:* El proyecto utiliza datos en formato JSON, bajados de MovieLens Tag Genome Dataset 2021, los datos incluyen:
-   *metadata.json*: Contiene información de películas como el elenco, directores y calificación promedio.
-   *ratings.json*: Contiene las calificaciones dadas por los usuarios a las películas en MovieLens.
-   *reviews.json*: Contiene reseñas de películas recopiladas de IMDB.
-   *survey_answers.json*: Incluye respuestas de usuarios a encuestas sobre la relevancia de ciertas etiquetas para diferentes películas.
-   *tag_count.json*: Contabiliza cuántas veces los usuarios han agregado etiquetas a las películas.
-   *tags.json*: Contiene la lista de etiquetas e identificadores utilizados en el dataset

## Links del DataSet utilizado
**Nota:** Los datos utilizados al ser tan extensos (32Millones) no es posible agregarlo en el repositorio, por lo que en el proyecto debera añadirse la carpeta **data/** en donde instalara los DataSets para su uso en el proyecto.

**Pagina del DataSet:** [MovieLens](https://grouplens.org/datasets/movielens/)

**Documentacion del DataSet** [Documentation](https://files.grouplens.org/datasets/tag-genome-2021/genome_2021_readme.txt)

# Se busca:
-   *Perfil de Usuario Personalizado:* construir un perfil de usuario basado en las películas valoradas previamente, para capturar las preferencias en términos de etiquetas relevantes.
-   *Similitud entre Películas:* identificar películas similares en términos de contenido en base a las etiquetas más relevantes.
-   *Recomendación Basada en Contenido:* generar recomendaciones  que coincidan con los intereses de contenido específicos de cada usuario.
