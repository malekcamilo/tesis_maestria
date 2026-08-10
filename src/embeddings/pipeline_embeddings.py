from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
MODEL_REVISION = "c4d86612f51b4f46759c8390d1798c2febe71b93"
N_COMPONENTS = 4

TOKENS = ["gato", "negro", "ratón", "blanco", "perro"]
POSITIONS = list(range(10))
CORPUS = [
    "el gato negro persigue al ratón blanco.",
    "el perro negro ladra al gato negro.",
    "el ratón blanco huye del perro.",
]
TARGET_WORD = "negro"

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIRECTORY / "pipeline_embeddings.log"


def configure_logger() -> logging.Logger:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("pipeline_embeddings")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(message)s")

    file_handler = logging.FileHandler(OUTPUT_PATH, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def single_token_id(tokenizer: Any, token: str) -> int:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"El ejemplo requiere que {token!r} sea un unico token; se obtuvieron {token_ids}"
        )
    return token_ids[0]


def make_record(
    vector_type: str,
    label: str,
    vector: np.ndarray,
    **metadata: Any,
) -> dict[str, Any]:
    return {
        "type": vector_type,
        "label": label,
        "vector": np.asarray(vector),
        "meta": metadata,
    }


def format_vector(vector: np.ndarray, decimals: int = 2) -> str:
    return "[" + ", ".join(f"{value:.{decimals}f}" for value in vector) + "]"


def build_vectors(model: Any, tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    d_model = model.config.hidden_size
    embeddings_matrix = model.embeddings.word_embeddings.weight.detach().cpu().numpy()
    position_embeddings = model.embeddings.position_embeddings.weight.detach().cpu().numpy()
    records: list[dict[str, Any]] = []

    # Paso 1: Token Embeddings.
    for token in TOKENS:
        token_id = single_token_id(tokenizer, token)
        records.append(make_record("TOKEN", token, embeddings_matrix[token_id]))

    # Paso 2: Positional Encodings aprendidos.
    for position in POSITIONS:
        records.append(
            make_record(
                "POSITIONAL",
                str(position),
                position_embeddings[position],
                position=position,
            )
        )

    # Paso 3: Composicion de entrada.
    target_id = single_token_id(tokenizer, TARGET_WORD)
    scaled_word_vector = embeddings_matrix[target_id] * math.sqrt(d_model)

    for document_index, sentence in enumerate(CORPUS, start=1):
        input_ids = tokenizer.encode(sentence, add_special_tokens=True)
        for position, token_id in enumerate(input_ids):
            if token_id != target_id:
                continue
            composed = scaled_word_vector + position_embeddings[position]
            records.append(
                make_record(
                    "INPUT",
                    f"{TARGET_WORD}_doc{document_index}_pos{position}",
                    composed,
                    token=TARGET_WORD,
                    document=document_index,
                    position=position,
                )
            )

    attention_self = model.encoder.layer[0].attention.self
    attention_output = model.encoder.layer[0].attention.output
    intermediate = model.encoder.layer[0].intermediate
    output_dense = model.encoder.layer[0].output.dense
    ffn_details: dict[str, Any] = {}

    # Pasos 4-7: Self-Attention, Add, Norm y FFN para cada token.
    for token in TOKENS:
        token_id = single_token_id(tokenizer, token)
        input_ids = torch.tensor([[token_id]], dtype=torch.long)

        with torch.no_grad():
            embedding_output = model.embeddings(input_ids)
            x_input = embedding_output[0, 0]

            self_attention_output = attention_self(embedding_output)[0]
            x_sublayer = attention_output.dense(self_attention_output)[0, 0]

            x_add = x_input + x_sublayer
            x_norm = attention_output.LayerNorm(x_add)

            pre_activation = intermediate.dense(x_norm.unsqueeze(0))
            post_activation = intermediate.intermediate_act_fn(pre_activation)
            ffn_output = output_dense(post_activation)[0]

        records.extend(
            [
                make_record(
                    "ATTENTION_INPUT",
                    f"{token}_Input",
                    x_input.cpu().numpy(),
                    token=token,
                ),
                make_record(
                    "SELF_ATTENTION",
                    f"{token}_SelfAttention",
                    x_sublayer.cpu().numpy(),
                    token=token,
                ),
                make_record(
                    "ADD",
                    f"{token}_Add",
                    x_add.cpu().numpy(),
                    token=token,
                ),
                make_record(
                    "NORM",
                    f"{token}_Norm",
                    x_norm.cpu().numpy(),
                    token=token,
                ),
                make_record(
                    "FFN",
                    f"{token}_FFN",
                    ffn_output.cpu().numpy(),
                    token=token,
                ),
            ]
        )

        ffn_details[token] = {
            "mean_add": x_add.mean().item(),
            "var_add": x_add.var(unbiased=False).item(),
            "mean_norm": x_norm.mean().item(),
            "var_norm": x_norm.var(unbiased=False).item(),
            "pre_activation": pre_activation[0].cpu().numpy(),
            "post_activation": post_activation[0].cpu().numpy(),
            "output": ffn_output.cpu().numpy(),
        }

    return records, ffn_details


def fit_and_project_records(records: list[dict[str, Any]]) -> PCA:
    """Ajusta un PCA comun y proyecta todos los vectores generados."""
    all_vectors = np.stack(
        [np.asarray(item["vector"], dtype=np.float64) for item in records]
    )
    pca = PCA(n_components=N_COMPONENTS, svd_solver="full")
    transformed_vectors = pca.fit_transform(all_vectors)

    for item, transformed_vector in zip(records, transformed_vectors, strict=True):
        item["vector_pca"] = transformed_vector

    return pca


def records_of_type(
    records: list[dict[str, Any]], vector_type: str
) -> list[dict[str, Any]]:
    return [item for item in records if item["type"] == vector_type]


def print_tables(
    logger: logging.Logger,
    records: list[dict[str, Any]],
    ffn_details: dict[str, Any],
    activation_name: str,
    pca: PCA,
) -> None:
    logger.info("=== SIETE TABLAS PEDAGOGICAS DEL CAPITULO 3 (PE APRENDIDOS) ===")
    logger.info("Modelo: %s", MODEL_NAME)
    logger.info("Revision: %s", MODEL_REVISION)
    logger.info("Vectores procesados: %d", len(records))
    logger.info("Componentes PCA: %d", N_COMPONENTS)
    logger.info("Activacion FFN: %s", activation_name)

    logger.info("\n>>> PASO 1 - TABLA: Token Embeddings")
    for item in records_of_type(records, "TOKEN"):
        logger.info(
            "Token: %-10s | Vec: %s", item["label"], format_vector(item["vector_pca"])
        )

    logger.info("\n>>> PASO 2 - TABLA: Positional Encodings")
    positional_records = {
        item["meta"]["position"]: item for item in records_of_type(records, "POSITIONAL")
    }
    for document, position in [(1, 3), (2, 3), (2, 8)]:
        item = positional_records[position]
        logger.info(
            "Token: %-6s | Doc: %d | Pos: %d | Vec: %s",
            TARGET_WORD,
            document,
            position,
            format_vector(item["vector_pca"]),
        )

    logger.info("\n>>> PASO 3 - TABLA: Composicion de Embeddings")
    for item in records_of_type(records, "INPUT"):
        metadata = item["meta"]
        logger.info(
            "Token: %-6s | Doc: %d | Pos: %d | Vec: %s",
            metadata["token"],
            metadata["document"],
            metadata["position"],
            format_vector(item["vector_pca"]),
        )

    logger.info("\n>>> PASO 4 - TABLA: Self-Attention")
    for item in records_of_type(records, "SELF_ATTENTION"):
        logger.info(
            "Token: %-10s | Vec: %s",
            item["meta"]["token"],
            format_vector(item["vector_pca"]),
        )

    logger.info("\n>>> PASO 5 - TABLA: Add")
    for item in records_of_type(records, "ADD"):
        logger.info(
            "Token: %-10s | Vec: %s",
            item["meta"]["token"],
            format_vector(item["vector_pca"]),
        )

    logger.info("\n>>> PASO 6 - TABLA: Norm")
    for item in records_of_type(records, "NORM"):
        logger.info(
            "Token: %-10s | Vec: %s",
            item["meta"]["token"],
            format_vector(item["vector_pca"]),
        )

    logger.info("\n>>> PASO 7 - TABLA: Feed Forward Network con GELU")
    for item in records_of_type(records, "FFN"):
        logger.info(
            "Token: %-10s | Vec: %s",
            item["meta"]["token"],
            format_vector(item["vector_pca"]),
        )

    example = ffn_details["gato"]
    logger.info("\n>>> DETALLE NORM PARA EL TOKEN gato EN EL ESPACIO ORIGINAL")
    logger.info("Media antes de Norm: %.4f", example["mean_add"])
    logger.info("Varianza antes de Norm: %.4f", example["var_add"])
    logger.info("Media despues de Norm: %.4f", example["mean_norm"])
    logger.info("Varianza despues de Norm: %.4f", example["var_norm"])

    logger.info("\n>>> DETALLE FFN PARA EL TOKEN gato EN EL ESPACIO ORIGINAL")
    logger.info(
        "Antes de GELU (primeros 5): %s",
        format_vector(example["pre_activation"][:5], 6),
    )
    logger.info(
        "Despues de GELU (primeros 5): %s",
        format_vector(example["post_activation"][:5], 6),
    )
    logger.info(
        "Salida FFN (primeros 5): %s",
        format_vector(example["output"][:5], 6),
    )

    logger.info(
        "\nVarianza explicada por el PCA: %s",
        format_vector(pca.explained_variance_ratio_, 8),
    )

def main() -> None:
    logger = configure_logger()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model.eval()

    records, ffn_details = build_vectors(model, tokenizer)

    expected_records = len(TOKENS) + len(POSITIONS) + 3 + 5 * len(TOKENS)
    if len(records) != expected_records:
        raise RuntimeError(
            f"Se esperaban {expected_records} vectores y se generaron {len(records)}"
        )

    pca = fit_and_project_records(records)
    print_tables(logger, records, ffn_details, str(model.config.hidden_act), pca)


if __name__ == "__main__":
    main()
