import logging
from tokenizers import Tokenizer, trainers, models, pre_tokenizers

# Configuración de Logs
logging.basicConfig(
    filename='wordpiece.log',
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
wordpiece_tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
wordpiece_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
wordpiece_trainer = trainers.WordPieceTrainer(
    vocab_size=45,
    min_frequency=1,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
)

# Entrenamiento
wordpiece_tokenizer.train_from_iterator(corpus, trainer=wordpiece_trainer)

# Vocabulario
vocab_wordpiece = wordpiece_tokenizer.get_vocab()
vocab_size = len(vocab_wordpiece)

logging.info(">>> vocab_size")
logging.info(vocab_size)

# Mostramos los primeros 10 items ordenados por ID para inspeccionar
logging.info(">>> vocab_wordpiece")
#sorted_vocab = sorted(vocab_wordpiece.items(), key=lambda x: x[1])
#logging.info(sorted_vocab[:10])
logging.info(vocab_wordpiece)

# Documento
doc = corpus[0]
output = wordpiece_tokenizer.encode(doc)

logging.info(">>> doc")
logging.info(doc)
logging.info(">>> output.tokens")
logging.info(output.tokens)
logging.info(">>> output.ids")
logging.info(output.ids)

#for doc in enumerate(corpus):
#    output = wordpiece_tokenizer.encode(doc)
    
#    logging.info(">>> doc")
#    logging.info(doc)

#    logging.info(">>> output.tokens")
#    logging.info(output.tokens)
    
#    logging.info(">>> output.ids")
#    logging.info(output.ids)
