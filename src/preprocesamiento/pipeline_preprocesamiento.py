import json
import logging
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizers.normalizers import Sequence, NFC, Replace, Strip
from tokenizers import Regex

# Configuración de Logs
logging.basicConfig(
    filename='pipeline_preprocesamiento.log',
    filemode='w',
    level=logging.INFO,
    format='%(message)s'
)

# Pipeline de normalización
normalizador = Sequence([
    NFC(),
    Replace(Regex(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]'), ""),
    Replace(Regex(r'\s+'), " "),
    Strip()
])

def normalizar_texto(texto):
    if not isinstance(texto, str):
        return texto
    return normalizador.normalize_str(texto)

def procesar_sedici():
    ruta_entrada = 'sedici.jsonl'
    
    patron_informatica = re.compile(r'informática|computación|software|sistemas de información|computer science', re.IGNORECASE)
    clases_reducir = [
        "Imagen en movimiento",
        "Reporte",
        "Documento institucional",
        "Publicacion seriada",
        "Proyecto",
        "Imagen fija",
        "Conjunto de datos",
        "Audio"
    ]
    
    registros_validos = []
    total_evaluados = 0
    total_filtrados = 0
    
    logging.info("Iniciando procesamiento de sedici.jsonl")
    
    with open(ruta_entrada, 'r', encoding='utf-8') as f:
        for linea in f:
            total_evaluados += 1
            registro = json.loads(linea)
            
            # Filtrado de informática
            es_informatica = False
            temas = registro.get('subjects', [])
            if any(patron_informatica.search(t) for t in temas):
                es_informatica = True
                
            if not es_informatica:
                continue
                
            # Extracción y Normalización de 3 columnas
            title = normalizar_texto(registro.get('title', ''))
            abstract = normalizar_texto(registro.get('abstract', ''))
            tipo = normalizar_texto(registro.get('type', ''))
            
            # Agrupación de clases
            if tipo in clases_reducir:
                tipo = 'Otro'
            if not tipo:
                tipo = 'Desconocido'
                
            registros_validos.append({
                'title': title,
                'abstract': abstract,
                'type': tipo
            })
            total_filtrados += 1
            
            if total_filtrados % 5000 == 0:
                logging.info(f"Registros filtrados extraídos: {total_filtrados}")
                
    logging.info(f"Total evaluados: {total_evaluados}")
    logging.info(f"Total informatica extraidos: {total_filtrados}")
    
    # Conversión a DataFrame para estratificación
    df = pd.DataFrame(registros_validos)
    df['type'] = df['type'].fillna('Desconocido')
    
    # Eliminación de clases con menos de 10 instancias
    frecuencias = df['type'].value_counts()
    clases_validas = frecuencias[frecuencias >= 10].index
    df_valido = df[df['type'].isin(clases_validas)]
    
    logging.info("Iniciando partición estratificada")
    
    # Primera división (70% train / 30% temp)
    df_train, df_temp = train_test_split(
        df_valido, 
        test_size=0.30, 
        # random_state=42,
        shuffle=True,
        stratify=df_valido['type']
    )
    
    # Validación clases temp
    frec_temp = df_temp['type'].value_counts()
    clases_temp_validas = frec_temp[frec_temp >= 2].index
    df_temp_valido = df_temp[df_temp['type'].isin(clases_temp_validas)]
    
    # Segunda división (15% val / 15% test)
    df_val, df_test = train_test_split(
        df_temp_valido, 
        test_size=0.50, 
        # random_state=42,
        shuffle=True,
        stratify=df_temp_valido['type']
    )
    
    # Exportación
    df_train.to_json('train_strat.jsonl', orient='records', lines=True, force_ascii=False)
    df_val.to_json('val_strat.jsonl', orient='records', lines=True, force_ascii=False)
    df_test.to_json('test_strat.jsonl', orient='records', lines=True, force_ascii=False)
    
    logging.info(f"Entrenamiento: {len(df_train)} registros")
    logging.info(f"Validación: {len(df_val)} registros")
    logging.info(f"Prueba: {len(df_test)} registros")
    
    print("Preprocesamiento completado. Revisa pipeline_preprocesamiento.log para más detalles.")
    print(f"Entrenamiento: {len(df_train)}")
    print(f"Validación: {len(df_val)}")
    print(f"Prueba: {len(df_test)}")

if __name__ == "__main__":
    procesar_sedici()
