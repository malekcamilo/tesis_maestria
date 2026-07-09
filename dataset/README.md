# Conjunto de Datos (`dataset/`)

Este directorio aloja las particiones de datos estructurados del repositorio SEDICI de la Universidad Nacional de La Plata (UNLP), filtrados para el dominio de Ciencias de la Computación, y utilizados en los procesos de entrenamiento, validación y prueba de los modelos adaptados en la tesis.

## Particiones y Volumen de Datos

El corpus original fue procesado (normalización NFC y colapso de espacios) y filtrado por dominio, reduciéndose a un total de **19 974 registros**. Estos datos se distribuyen mediante partición estratificada (preservando la proporción de clases de la variable objetivo) en formato JSON Lines (`.jsonl`):

*   **`train_tesis.jsonl`**: Partición de entrenamiento que representa el **70 %** del corpus total (**13 981 registros**), empleada para la optimización de parámetros y cálculo del gradiente en el ajuste fino.
*   **`val_tesis.jsonl`**: Partición de validación que representa el **15 %** del corpus total (**2 996 registros**), utilizada para el monitoreo del entrenamiento, ajuste de hiperparámetros y detección temprana de sobreajuste.
*   **`test_tesis.jsonl`**: Partición de prueba independiente que representa el **15 %** del corpus total (**2 997 registros**), reservada exclusivamente para la evaluación final y obtención de las métricas de generalización del modelo.

## Esquema de Datos

Cada registro dentro de los archivos `.jsonl` cuenta con una estructura plana de tres campos clave:
*   `title` (String): Título normalizado del documento.
*   `abstract` (String): Resumen o abstract estructurado del recurso.
*   `type` (String): Etiqueta objetivo (clase) que denota la tipología documental.

### Ejemplo de Registro:
```json
{
  "title": "Verificación del hablante mediante dispositivos móviles en entornos ruidosos",
  "abstract": "El interés por las aplicaciones biométricas ha crecido...",
  "type": "Objeto de conferencia"
}
```

## Clases del Dominio

La variable objetivo `type` se clasifica en una de las seis categorías de tipologías documentales consolidadas:
1.  `Objeto de conferencia`
2.  `Artículo`
3.  `Tesis`
4.  `Libro`
5.  `Objeto de aprendizaje`
6.  `Otro` (unificación de categorías minoritarias del repositorio original)
