import logging
from tokenizers import pre_tokenizers

# Configuración de Logs
logging.basicConfig(
    filename='pretokenizador.log',
    filemode='w',
    level=logging.INFO,
    format='%(message)s'
)

corpus = [
    "el gato negro persigue al raton blanco.",
    "el perro negro ladra al gato negro.",
    "el raton blanco huye del perro."
]

# Configuración
pretok_whitespace = pre_tokenizers.Whitespace()

# Ejecución
doc = corpus[0]
resultado = pretok_whitespace.pre_tokenize_str(doc)
tokens_list = [token for token, _ in resultado]

logging.info(">>> doc")
logging.info(doc)

logging.info(">>> tokens_list")
logging.info(tokens_list)
