import logging
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.decomposition import PCA
import numpy as np

# Configuración de Logs
logging.basicConfig(
    filename='token_embeddings.log',
    filemode='w',
    level=logging.INFO,
    format='%(message)s'
)

# Cargar Modelo
model_name = 'dccuchile/bert-base-spanish-wwm-cased'
try:
    logging.info(f"Cargando modelo: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
except OSError:
    logging.error("No se pudo descargar el modelo.")
    exit()

# Datos
tokens_interes = ["gato", "negro", "ratón", "blanco", "perro"]

# Obtener embeddings (768d)
embeddings_matrix = model.embeddings.word_embeddings.weight.detach().numpy()
vectors_768d = []
tokens_found = []

for token in tokens_interes:
    # Usamos el primer ID si se tokeniza en varios
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if not token_ids:
        continue
    
    idx = token_ids[0]
    vector = embeddings_matrix[idx]
    vectors_768d.append(vector)
    tokens_found.append(token)

# Reducción PCA (768d -> 4d)
pca = PCA(n_components=4)
vectors_4d = pca.fit_transform(np.array(vectors_768d))

# Imprimir Resultados
logging.info(">>> Tokens y Vectores (PCA 4d)")
for token, vec in zip(tokens_found, vectors_4d):
    logging.info(f"Token: {token}")
    vec_str = "[" + ", ".join([f"{x:.2f}" for x in vec]) + "]"
    logging.info(vec_str)
