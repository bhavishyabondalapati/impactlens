"""all-MiniLM-L6-v2 as ONNX, so retrieval runs in AWS Lambda without PyTorch.

sentence-transformers needs PyTorch (~700 MB), which is too heavy for a Lambda image. The model
itself is a small BERT (22 M parameters). We export it once to ONNX on the laptop and do the
two steps sentence-transformers adds on top of BERT ourselves, in numpy:

  1. mean pooling: average the token vectors, ignoring padding (attention mask = 0)
  2. L2 normalisation, so inner product = cosine similarity (what the FAISS index expects)

The tokenizer is the model's own ``tokenizer.json`` run by the Rust ``tokenizers`` library
(no PyTorch either). tests/test_aws.py checks that the ONNX vectors match sentence-transformers
and that FAISS returns the same SOP excerpts with either embedder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from impactlens.agents.rag import EMBED_MODEL

EMBEDDER_DIR = Path("build/embedder")
MAX_TOKENS = 256  # all-MiniLM-L6-v2's max_seq_length: longer text is truncated, as in training


def export_minilm(out_dir: Path = EMBEDDER_DIR, model_name: str = EMBED_MODEL) -> Path:
    """Write ``minilm.onnx`` and ``tokenizer.json`` to ``out_dir`` (needs torch: laptop only)."""
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel

    out_dir.mkdir(parents=True, exist_ok=True)
    bert = AutoModel.from_pretrained(model_name).eval()

    class Wrapper(torch.nn.Module):  # passes the inputs by name (forward's order has changed)
        def __init__(self):
            super().__init__()
            self.bert = bert

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.bert(input_ids=input_ids, attention_mask=attention_mask,
                             token_type_ids=token_type_ids).last_hidden_state  # fmt: skip

    model = Wrapper().eval()
    dummy = {
        "input_ids": torch.ones(2, 16, dtype=torch.long),
        "attention_mask": torch.ones(2, 16, dtype=torch.long),
        "token_type_ids": torch.zeros(2, 16, dtype=torch.long),
    }
    names = list(dummy)
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
        str(out_dir / "minilm.onnx"),
        input_names=names,
        output_names=["last_hidden_state"],
        # batch size and sequence length change from call to call
        dynamic_axes={n: {0: "batch", 1: "tokens"} for n in [*names, "last_hidden_state"]},
        opset_version=17,
        dynamo=False,  # the classic exporter: simple, and enough for a plain BERT
    )
    tok = hf_hub_download(model_name, "tokenizer.json")
    (out_dir / "tokenizer.json").write_bytes(Path(tok).read_bytes())
    return out_dir


class OnnxEmbedder:
    """Drop-in for rag.Embedder: texts -> (n, 384) float32 unit vectors."""

    def __init__(self, model_dir: Path = EMBEDDER_DIR):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self.tokenizer.enable_truncation(MAX_TOKENS)
        self.tokenizer.enable_padding()  # pad each batch to its longest text
        self.session = ort.InferenceSession(str(Path(model_dir) / "minilm.onnx"),
                                            providers=["CPUExecutionProvider"])  # fmt: skip

    def __call__(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer.encode_batch(texts[i : i + batch_size])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            types = np.array([e.type_ids for e in enc], dtype=np.int64)
            hidden = self.session.run(None, {"input_ids": ids, "attention_mask": mask,
                                             "token_type_ids": types})[0]  # fmt: skip
            m = mask[:, :, None].astype(np.float32)
            pooled = (hidden * m).sum(1) / np.maximum(m.sum(1), 1e-9)  # mean over real tokens
            out.append(pooled / np.linalg.norm(pooled, axis=1, keepdims=True))
        return np.concatenate(out).astype(np.float32)
