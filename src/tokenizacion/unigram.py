import logging
from tokenizers import Tokenizer, trainers, models, pre_tokenizers

# Configuración de Logs
logging.basicConfig(
    filename='unigram.log',
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
unigram_tokenizer = Tokenizer(models.Unigram())
unigram_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
unigram_trainer = trainers.UnigramTrainer(
    vocab_size=45,
    show_progress=False,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
    unk_token="[UNK]"
)

# Entrenamiento
unigram_tokenizer.train_from_iterator(corpus, trainer=unigram_trainer)

# Vocabulario
vocab_unigram = unigram_tokenizer.get_vocab()
vocab_size = len(vocab_unigram)

logging.info(">>> vocab_size")
logging.info(vocab_size)

logging.info(">>> vocab_unigram")
logging.info(vocab_unigram)
#sorted_vocab = dict(sorted(vocab_unigram.items(), key=lambda x: x[1]))
#logging.info(sorted_vocab)

# Documento
doc = corpus[0]
output = unigram_tokenizer.encode(doc)

logging.info(">>> doc")
logging.info(doc)

logging.info(">>> output.tokens")
logging.info(output.tokens)

logging.info(">>> output.ids")
logging.info(output.ids)
