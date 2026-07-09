# Código Fuente (`src/`)

Este directorio contiene los scripts responsables del *pipeline* de preprocesamiento de datos, entrenamiento y evaluación para la tesis de maestría.
Como así también el código utilizado para las implementaciones en el estado del arte de la tesis, según lo solicitado por la comisión evaluadora.

## Estructura del Directorio

El código fuente está organizado en los siguientes subdirectorios funcionales:

*   **`experimentos/`**: Contiene los scripts principales para la ejecución de los experimentos de *fine-tuning*. Destaca un único script de entrenamiento y evaluación que ejecuta los diferentes métodos de adaptación eficiente.
*   **`slurm/`**: *Scripts* de tipo bash (`.sh`) encargados de orquestar la ejecución de los experimentos en la supercomputadora Clementina XXI mediante el sistema de colas Slurm (e.j., `sbatch`).
*   **`preprocesamiento/`**: *Scripts* dedicados a la limpieza, transformación y formateo de los datos recolectados en su estructura final de partición.
*   **`scraping/`**: *Scripts* orientados a la extracción de datos del repositorio SEDICI.
*   **`embeddings/`**: *Scripts* didácticos y experimentales utilizados en el Capítulo 3 para ilustrar la inicialización de *token embeddings*, codificaciones posicionales (*positional encodings*) y la composición/normalización en el bloque Transformer.
*   **`tokenizacion/`**: *Scripts* demostrativos empleados en el Capítulo 3 para ejemplificar el *pipeline* de tokenización moderna (normalización, pretokenización y algoritmos de subpalabras como BPE, WordPiece y Unigram).
