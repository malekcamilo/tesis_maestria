import logging
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Configuración de Logs
logging.basicConfig(
    filename='ffn_add_norm.log',
    filemode='w',
    level=logging.INFO,
    format='%(message)s'
)

# Configuración Modelo
model_name = 'dccuchile/bert-base-spanish-wwm-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Desactivar gradientes para inferencia
model.eval()

# Palabra de ejemplo
word = "gato"
input_ids = torch.tensor(tokenizer.encode(word, add_special_tokens=False))
token_id = input_ids[0]

# Simulación de Forward Pass hasta Layer 0 Output
# Obtenemos los valores reales de BERT para ilustrar Add (Residual) y Norm por separado

logging.info(f"--- Análisis para el token: '{word}' en Layer 0 ---")

# Input Embeddings (Salida de la capa de embeddings = Entrada al Encoder)
embedding_output = model.embeddings(input_ids.unsqueeze(0))
x_input = embedding_output[0, 0].detach() # Vector del token [768]

logging.info(f"1. Entrada (x):")
logging.info(f"   Primeros 5 valores: {x_input[:5].numpy()}")
logging.info(f"   Media={x_input.mean().item():.4f}, Var={x_input.var().item():.4f}")

# Attention Output (Sublayer)
# Pasamos por el módulo de Self-Attention
# layer 0 -> attention -> self
attention_self = model.encoder.layer[0].attention.self
# layer 0 -> attention -> output (donde está la densa de proyección, dropout y layernorm)
attention_output_layer = model.encoder.layer[0].attention.output

with torch.no_grad():
    # Self-Attention (Q, K, V) -> Context Vectors
    # Necesita mask (dummy de 1s), encoder_hidden_states=None
    head_mask = [None] * model.config.num_hidden_layers
    extended_attention_mask = model.get_extended_attention_mask(torch.ones(1, 1), (1, 1), torch.device('cpu'))
    
    self_output = attention_self(embedding_output)[0] # [1, 1, 768]
    
    # Proyección Densa (W_o) antes de sumar
    # attention_output_layer tiene: dense, LayerNorm, dropout
    # Queremos ver el "Sublayer(x)" puro (la salida de la atención proyectada)
    dense_output = attention_output_layer.dense(self_output)
    x_sublayer = dense_output[0, 0]
    
    logging.info(f"2. Salida Subcapa (Attention(x)):")
    logging.info(f"   Primeros 5 valores: {x_sublayer[:5].numpy()}")
    
    # Conexión Residual (Add = x + Sublayer(x))
    x_add = x_input + x_sublayer
    
    logging.info(f"3. Resultado Suma (x + Sublayer(x)):")
    logging.info(f"   Primeros 5 valores: {x_add[:5].numpy()}")
    logging.info(f"   Media={x_add.mean().item():.4f}, Var={x_add.var().item():.4f}")
    
    # Normalización (Norm(Add))
    x_norm = attention_output_layer.LayerNorm(x_add)
    
    logging.info(f"4. Resultado Normalización (LayerNorm):")
    logging.info(f"   Primeros 5 valores: {x_norm[:5].numpy()}")
    logging.info(f"   Media={x_norm.mean().item():.4f}, Var={x_norm.var().item():.4f}")
    logging.info("")


# Simulación de Feed Forward Network (FFN)
# Ecuación: FFN(x) = GELU(xW1 + b1)W2 + b2
d_model = 768
d_ff = 3072

# Pesos reales de la capa 0 de BERT
layer_0_ffn = model.encoder.layer[0].intermediate.dense
layer_0_output = model.encoder.layer[0].output.dense

# Proyección a d_ff (Expansión)
with torch.no_grad():
    # Proyección lineal
    hidden = layer_0_ffn(x_input.unsqueeze(0)) # unsqueeze para dimensión batch
    
    logging.info("--- 2. Feed Forward Network (FFN) ---")
    logging.info(f"Expansión a dimensión intermedia (d_ff): {hidden.shape[1]}")
    logging.info(f"Valores intermedios (tras GELU): {hidden[0][:5].numpy()}")
    
    # Proyección a d_model (Compresión)
    output = layer_0_output(hidden)
    
    logging.info(f"Proyección de vuelta a d_model: {output.shape[1]}")
    logging.info(f"Valores finales: {output[0][:5].numpy()}")
