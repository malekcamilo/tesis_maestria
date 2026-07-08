import numpy as np
import math
import logging
from transformers import AutoTokenizer
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine

# Configuración de Logs
logging.basicConfig(
    filename='positional_encoding.log',
    filemode='w',
    level=logging.INFO,
    format='%(message)s'
)

# Configuración
model_name = "dccuchile/bert-base-spanish-wwm-cased"
d_model = 768

# Corpus
corpus = [
    "el gato negro persigue al ratón blanco.",
    "el perro negro ladra al gato negro.",
    "el ratón blanco huye del perro."
]

# Inicializar Tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
except Exception as e:
    logging.error(f"Error cargando tokenizer: {e}")
    exit()

# Función de Positional Encoding (Seno/Coseno)
def get_positional_encoding(pos, d_model):
    pe = np.zeros(d_model)
    for i in range(0, d_model, 2):
        div_term = math.exp(i * -math.log(10000.0) / d_model)
        pe[i] = math.sin(pos * div_term)
        if i + 1 < d_model:
            pe[i + 1] = math.cos(pos * div_term)
    return pe

# Encontrar posiciones de "negro" en el corpus
target_word = "negro"
occurrences = []

logging.info(f"Analizando invariancia de Positional Encoding para: '{target_word}'")

target_ids = tokenizer.encode(target_word, add_special_tokens=False)
if not target_ids:
    exit()
target_id = target_ids[0]

for sent_idx, sentence in enumerate(corpus):
    input_ids = tokenizer.encode(sentence, add_special_tokens=True)
    
    for pos, tid in enumerate(input_ids):
        if tid == target_id:
            pe_vector = get_positional_encoding(pos, d_model)
            occurrences.append({
                "doc_id": sent_idx,
                "pos": pos,
                "vector": pe_vector
            })

# Análisis de Similitud (Demostración de Independencia del Contexto)
logging.info(">>> Verificación de Similitud")

# Comparar (Doc 1, Pos 3) vs (Doc 2, Pos 3)
vec_a = occurrences[0]["vector"] # Doc 1, Pos 3
vec_b = occurrences[1]["vector"] # Doc 2, Pos 3
# Comparar (Doc 2, Pos 3) vs (Doc 2, Pos 8)
vec_c = occurrences[2]["vector"] # Doc 2, Pos 8

sim_ab = 1 - cosine(vec_a, vec_b)
sim_bc = 1 - cosine(vec_b, vec_c)

logging.info(f"Similitud (Doc 1, Pos 3) vs (Doc 2, Pos 3): {sim_ab:.5f} (Misma posición, distinto documento)")
logging.info(f"Similitud (Doc 2, Pos 3) vs (Doc 2, Pos 8): {sim_bc:.5f} (Mismo documento, distinta posición)")


# Visualización (PCA)
logging.info("-" * 20)
logging.info(f">>> Vectores PE (PCA 4d)")

# Entrenar PCA con rango base para consistencia
base_vectors = [get_positional_encoding(p, d_model) for p in range(10)]
pca = PCA(n_components=4)
pca.fit(base_vectors)

for item in occurrences:
    vec_pca = pca.transform([item["vector"]])[0]
    vec_str = "[" + ", ".join([f"{x:.2f}" for x in vec_pca]) + "]"
    logging.info(f"Doc: {item['doc_id'] + 1} | Pos: {item['pos']} -> {vec_str}")


