# Métricas Consolidadas (`metricas/`)

Este directorio actúa como un repositorio de persistencia para todos los resultados consolidados de las etapas de entrenamiento y evaluación de los experimentos de la tesis de maestría.

## Formato de Almacenamiento

Los resultados están estructurados de forma tabular en archivos CSV (`.csv`) que agregan las corridas experimentales calculando el promedio y la desviación estándar. Las columnas segregan las variables de hiperconfiguración experimental:
*   `method`: Algoritmo de adaptación eficiente (e.j., `dora`, `lora`, `galore`).
*   `lora_r` / `galore_rank`: Rango/dimensión de la aproximación de baja descomposición.
*   `update_proj_gap`: Frecuencia de actualización de proyección (para GaLore).
*   `target_modules`: Módulos del Transformer objetivo (e.j., `all`, `attn`).

## Archivos Consolidados

El directorio contiene cuatro archivos con alcances específicos:

1.  **`metricas_globales_consolidadas.csv`**: Registra los recursos computacionales y estadísticas globales del entrenamiento.
    *   *Métricas clave*: Tiempos de entrenamiento (`train_time_hours`), consumo de VRAM (`base_memory_gb`, `peak_memory_gb`, `peak_reserved_memory_gb`), parámetros entrenables (`trainable_parameters`, `trainable_percent`), velocidad de procesamiento (`train_samples_per_second`, `train_steps_per_second`) y pérdidas finales de entrenamiento y evaluación (`final_train_loss`, `final_eval_loss`).
2.  **`metricas_eval_globales_consolidadas.csv`**: Consolida el rendimiento final del modelo y latencia en inferencia.
    *   *Métricas clave*: Exactitud (`accuracy`), macro-F1 (`macro_f1`), coincidencia exacta (`exact_match`) y velocidades de decodificación en inferencia (`inference_time_seconds`, `samples_per_second`, `tokens_per_second`).
3.  **`metricas_eval_clases_consolidadas.csv`**: Desagrega las métricas de rendimiento por categoría/clase en el corpus evaluado.
    *   *Métricas clave*: F1 *score* (`f1_score`), coincidencia exacta (`exact_match`), precisión (`precision`), exhaustividad (`recall`) y soporte de muestras (`support`), todos desglosados por clase (e.j., `articulo`, `libro`, `objeto de aprendizaje`, `objeto de conferencia`, `tesis`, `otro`).
4.  **`metricas_epocas_consolidadas.csv`**: Registra la evolución detallada paso a paso e historial por época durante el proceso de entrenamiento.
    *   *Métricas clave*: Pérdida por época (`loss`), norma de gradiente (`grad_norm`), tasa de aprendizaje (`learning_rate`), entropía (`entropy`), cantidad de tokens procesados (`num_tokens`), exactitud de tokens (`mean_token_accuracy`) y sus homólogos correspondientes para el conjunto de evaluación.
