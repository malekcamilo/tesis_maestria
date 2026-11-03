# Métricas Consolidadas DDP (`metricas/`)

Este directorio reúne los resultados consolidados del entrenamiento distribuido (DDP) de LLaMA 3.1 8B con DoRA `r = 32` (`all`) en Clementina XXI, presentados en el póster de las Jornadas UNLu 2026. Sigue el mismo formato que las métricas consolidadas de la tesis de maestría.

## Formato de Almacenamiento

Los resultados están estructurados de forma tabular en archivos CSV (`.csv`) que agregan las corridas experimentales calculando el promedio (`_mean`) y la desviación estándar (`_std`) entre semillas (`42`, `731558364`, `3688191069`). Las columnas segregan las variables de hiperconfiguración experimental:
*   `method`: Algoritmo de adaptación eficiente (`dora`).
*   `lora_r` / `galore_rank`: Rango/dimensión de la aproximación de baja descomposición (`galore_rank` queda vacío; se conserva por compatibilidad con la tesis).
*   `update_proj_gap`: Frecuencia de actualización de proyección (vacío; ídem).
*   `target_modules`: Módulos del Transformer objetivo (`all`).
*   `n_nodes`: Cantidad de nodos.
*   `g_gpus_per_node`: GPUs (tiles de Intel Max 1550) por nodo.
*   `world_size`: Total de GPUs (`n_nodes × g_gpus_per_node`).
*   `global_batch_size`: Lote global (8 por GPU × `world_size`).

## Configuraciones y Corridas

| `n_nodes` | `g_gpus_per_node` | `world_size` | `global_batch_size` | Entrenamiento | Evaluación |
|---|---|---|---|---|---|
| 1 | 1 | 1 | 8 | 3 semillas (tesis / JCST) | 3 semillas (tesis / JCST) |
| 1 | 4 | 4 | 32 | 3 semillas | 3 semillas |
| 2 | 4 | 8 | 64 | 3 semillas | 3 semillas |

La referencia de 1 GPU corresponde a las corridas de DoRA `r = 32` (`all`) de la tesis de maestría y del artículo enviado a JCST (mismos valores que en sus métricas consolidadas). Esas corridas usaron otro script, que no mide tiempo por paso ni comunicación, por lo que las columnas `step_time_*` y `comunicacion_*` quedan vacías en esa fila.

La evaluación se realiza siempre en 1 GPU sobre la partición de test (n = 2.997), independientemente de la topología de entrenamiento.

## Archivos Consolidados

El directorio contiene cuatro archivos con alcances específicos:

1.  **`metricas_globales_consolidadas.csv`**: Registra los recursos computacionales y estadísticas globales del entrenamiento.
    *   *Métricas clave*: Tiempos de entrenamiento (`train_time_hours`), consumo de VRAM por GPU (`base_memory_gb`, `peak_memory_gb`, `peak_reserved_memory_gb`), parámetros entrenables (`trainable_parameters`, `trainable_percent`), velocidad de procesamiento (`train_samples_per_second`, `train_steps_per_second`) y pérdidas finales de entrenamiento y evaluación (`final_train_loss`, `final_eval_loss`).
    *   *Tiempo por paso*: `step_time_steps_medidos`, `step_time_mediana_s`, `step_time_media_s`, `step_time_p90_s` (rank 0) y `step_time_mediana_max_entre_ranks_s` (rank más lento).
    *   *Comunicación* (solo con más de una GPU): tamaño de los gradientes (`comunicacion_gradientes_mb`, `comunicacion_buckets`), tiempo de un allreduce completo sin solapamiento (`comunicacion_allreduce_por_paso_s`), forward+backward con y sin sincronización (`comunicacion_fwd_bwd_con_sync_s`, `comunicacion_fwd_bwd_sin_sync_s`), overhead efectivo que queda en el paso (`comunicacion_expuesta_s`) y ambos como fracción del paso (`comunicacion_allreduce_fraccion_del_paso`, `comunicacion_expuesta_fraccion_del_paso`).
2.  **`metricas_eval_globales_consolidadas.csv`**: Consolida el rendimiento final del modelo y latencia en inferencia.
    *   *Métricas clave*: Exactitud (`accuracy`), macro-F1 (`macro_f1`), coincidencia exacta (`exact_match`) y velocidades de decodificación en inferencia (`inference_time_seconds`, `samples_per_second`, `tokens_per_second`).
3.  **`metricas_eval_clases_consolidadas.csv`**: Desagrega las métricas de rendimiento por categoría/clase en el corpus evaluado.
    *   *Métricas clave*: F1 *score* (`f1_score`), coincidencia exacta (`exact_match`), precisión (`precision`), exhaustividad (`recall`) y soporte de muestras (`support`), todos desglosados por clase (`articulo`, `libro`, `objeto de aprendizaje`, `objeto de conferencia`, `tesis`, `otro`).
4.  **`metricas_epocas_consolidadas.csv`**: Registra el historial por época durante el proceso de entrenamiento.
    *   *Métricas clave*: Pérdida por época (`loss`), norma de gradiente (`grad_norm`), tasa de aprendizaje (`learning_rate`), entropía (`entropy`), cantidad de tokens procesados (`num_tokens`), exactitud de tokens (`mean_token_accuracy`) y sus homólogos correspondientes para el conjunto de validación (`eval_*`).

### Dependencias del Entorno (Clementina)

- **PyTorch (Intel XPU):** `torch==2.11.0+xpu`
- **Modelos y NLP:** `transformers==5.5.0`, `peft==0.18.1`, `trl==1.0.0`, `accelerate==1.13.0`
- **Cuantización:** `bitsandbytes==0.49.2`
- **Datos:** `datasets==4.8.4`
- **Comunicación:** Intel MPI 2021.17.2, oneCCL 2021.17.2, libfabric 2.2.0 (`FI_PROVIDER=psm3`)
