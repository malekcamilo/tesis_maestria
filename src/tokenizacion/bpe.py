import logging
from tokenizers import Tokenizer, trainers, models, pre_tokenizers

# Configuración de Logs
logging.basicConfig(
    filename='bpe.log',
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
bpe_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
bpe_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
bpe_trainer = trainers.BpeTrainer(
    vocab_size=45, 
    min_frequency=1,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
)

# Entrenamiento
bpe_tokenizer.train_from_iterator(corpus, trainer=bpe_trainer)

# Vocabulario
vocab_bpe = bpe_tokenizer.get_vocab()
vocab_size = len(vocab_bpe)

logging.info(">>> vocab_size")
logging.info(vocab_size)

logging.info(">>> vocab_bpe")
logging.info(vocab_bpe)
#sorted_vocab = dict(sorted(vocab_bpe.items(), key=lambda x: x[1]))
#logging.info(sorted_vocab)

# Documento
doc = corpus[0]
output = bpe_tokenizer.encode(doc)

logging.info(">>> doc")
logging.info(doc)

logging.info(">>> output.tokens")
logging.info(output.tokens)

logging.info(">>> output.ids")
logging.info(output.ids)
