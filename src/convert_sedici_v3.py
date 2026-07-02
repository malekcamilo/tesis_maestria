import pandas as pd
import json
import logging
import re

logging.basicConfig(
    filename='convert_sedici_v3.log',
    filemode='w',
    level=logging.INFO,
    format='%(message)s'
)

def limpiar_texto(texto):
    if pd.isna(texto) or texto is None:
        return ""
    texto_str = str(texto)
    texto_str = re.sub(r'[\n\r\t]+', ' ', texto_str)
    texto_str = re.sub(r'\s+', ' ', texto_str)
    return texto_str.strip()

def extraer_lista(texto, separador='||'):
    if pd.isna(texto) or texto is None:
        return []
    elementos = str(texto).split(separador)
    return [limpiar_texto(e) for e in elementos if limpiar_texto(e)]

def obtener_mejor_valor(fila, columnas, idiomas_prioridad=['[es]', '[en]', '[pt]', '[fr]', '[de]', '[it]']):
    for idioma in idiomas_prioridad:
        for col in columnas:
            if idioma in col:
                val = limpiar_texto(fila.get(col, ''))
                if val:
                    return val
    
    for col in columnas:
        val = limpiar_texto(fila.get(col, ''))
        if val:
            return val
    return ""

def procesar_csv(ruta_csv, ruta_jsonl):
    logging.info(f"Iniciando procesamiento de {ruta_csv}")
    
    columnas_csv = pd.read_csv(ruta_csv, nrows=0).columns.tolist()
    
    def buscar_cols(patrones):
        return [c for c in columnas_csv if any(p in c for p in patrones)]

    cols_title = buscar_cols(['dc.title['])
    cols_abstract = buscar_cols(['abstract['])
    cols_subject = buscar_cols(['subject[', 'subject.ford[', 'subject.materias['])
    cols_type = buscar_cols(['type[', 'subtype['])
    
    columnas_usar = list(set(
        cols_title + cols_abstract + cols_subject + cols_type
    ))
    
    total_leidos = 0
    total_validos = 0
    chunksize = 10000
    
    with open(ruta_jsonl, 'w', encoding='utf-8') as f_out:
        for chunk in pd.read_csv(ruta_csv, usecols=columnas_usar, chunksize=chunksize, dtype=str):
            for _, row in chunk.iterrows():
                total_leidos += 1
                
                titulo_principal = obtener_mejor_valor(row, cols_title)
                abstract = obtener_mejor_valor(row, cols_abstract)
                
                if not titulo_principal:
                    continue
                
                temas = []
                for c in cols_subject:
                    temas.extend(extraer_lista(row.get(c, '')))
                temas_unicos = list(dict.fromkeys(temas))
                        
                registro = {
                    "title": titulo_principal,
                    "abstract": abstract,
                    "type": obtener_mejor_valor(row, cols_type),
                    "subjects": temas_unicos
                }
                
                f_out.write(json.dumps(registro, ensure_ascii=False) + '\n')
                total_validos += 1
                
            logging.info(f"Leídos: {total_leidos} | Válidos: {total_validos}")

    logging.info("Procesamiento finalizado.")
    logging.info(f"Total procesados: {total_leidos}")
    logging.info(f"Total convertidos: {total_validos}")

if __name__ == '__main__':
    csv_path = 'export_sedici_utf8.csv'
    jsonl_path = 'sedici.jsonl'
    procesar_csv(csv_path, jsonl_path)
