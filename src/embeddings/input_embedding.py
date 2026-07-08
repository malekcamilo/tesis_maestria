import logging
import math
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA

# Configuración de Logs
logging.basicConfig(
    filename='input_embedding.log',
    filemode='w',
    level=logging.INFO,
    format='%(message)s'
)

# Configuración Modelo
model_name = 'dccuchile/bert-base-spanish-wwm-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
d_model = model.config.hidden_size # 768

# Corpus y Target
corpus = [
    "el gato negro persigue al ratón blanco.",
    "el perro negro ladra al gato negro.",
    "el ratón blanco huye del perro."
]
target_word = "negro"

# Funciones Auxiliares
def get_positional_encoding(pos, d_model):
    pe = np.zeros(d_model)
    for i in range(0, d_model, 2):
        div_term = math.exp(i * -math.log(10000.0) / d_model)
        pe[i] = math.sin(pos * div_term)
        if i + 1 < d_model:
            pe[i + 1] = math.cos(pos * div_term)
    return pe

# Obtener Vector Base (Token Embedding)
# Obtenemos el vector genérico de "negro" de la matriz de embeddings
ids = tokenizer.encode(target_word, add_special_tokens=False)
token_id = ids[0]
embeddings_matrix = model.embeddings.word_embeddings.weight.detach().numpy()
base_vec = embeddings_matrix[token_id]

# Procesar Ocurrencias en el Corpus
input_data = []

logging.info(f"Procesando palabra: '{target_word}'")

for sent_idx, sentence in enumerate(corpus):
    input_ids = tokenizer.encode(sentence, add_special_tokens=True)
    
    # Buscar ocurrencias del token_id en esta oración
    for pos, tid in enumerate(input_ids):
        if tid == token_id:
            # Cálculo del Input Embedding: (Token * sqrt(d)) + PE
            scaled_vec = base_vec * math.sqrt(d_model)
            pe_vec = get_positional_encoding(pos, d_model)
            final_vec = scaled_vec + pe_vec
            
            input_data.append({
                "sent_idx": sent_idx,
                "pos": pos,
                "vector": final_vec,
                "context": sentence
            })

# PCA para visualización (768 -> 4)
if input_data:
    all_vectors = [item["vector"] for item in input_data]
    n_components = min(len(input_data), 4)
    pca = PCA(n_components=n_components)
    all_vectors_4d = pca.fit_transform(np.array(all_vectors))

    logging.info(f">>> Ocurrencias y Vectores Input (PCA {n_components}d)")
    for i, item in enumerate(input_data):
        vec_4d = all_vectors_4d[i]
        vec_str = "[" + ", ".join([f"{x:.2f}" for x in vec_4d]) + "]"
        
        logging.info(f"Doc: {item['sent_idx']} | Pos: {item['pos']}")
        logging.info(vec_str)
else:
    logging.info("No se encontraron ocurrencias.")
