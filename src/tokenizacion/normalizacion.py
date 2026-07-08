import logging
from tokenizers import normalizers

# Configuración de Logs
logging.basicConfig(
    filename='normalizacion.log',
    filemode='w',
    level=logging.INFO,
    format='%(message)s'
)

corpus = [
    "El gato NEGRO persigue al ratón blanco.",
    "El perro ladra al gato negro.",
    "El ratón blanco huye del gato."
]

# Pipeline normalización
normalizer_completo = normalizers.Sequence([
    normalizers.NFD(),           # Descomposición Unicode
    normalizers.Lowercase(),     # Minúsculas
    normalizers.StripAccents()   # Sin acentos
])

# Ejecución
doc = corpus[0]
doc_normalizado = normalizer_completo.normalize_str(doc)
    
logging.info(">>> doc")
logging.info(doc)
    
logging.info(">>> doc_normalizado")
logging.info(doc_normalizado)
