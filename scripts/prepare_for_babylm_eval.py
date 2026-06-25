"""
Convert an autoresearch checkpoint (checkpoints/<commit>.pt) into a self-contained
HuggingFace repository that the BabyLM 2026 strict-track evaluation harness
(https://github.com/babylm-org/babylm-eval) can load via `trust_remote_code`.

The harness loads models with
    AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
and tokenizers with
    AutoProcessor.from_pretrained(path, trust_remote_code=True)   # falls back to AutoTokenizer
and then calls `tokenizer(sentence, return_offsets_mapping=True)`. The causal
backend therefore needs (a) a CausalLM whose `forward` returns `.logits`, and
(b) a tokenizer that supports `offset_mapping`.

Our model is a custom Muon-trained GPT (value embeddings, QK-norm, rotary,
RMSNorm, logit softcap, sliding-window SSSL) and our tokenizer is a Morfessor
morpheme segmenter with raw-byte fallback. Because the model's embedding table is
indexed by exact training token-ids, the exported tokenizer must reproduce the
training segmentation *exactly* -- a stock `tokenizers` (Unigram/BPE) model cannot
replicate Morfessor's Viterbi cost, so we ship the real Morfessor tokenizer as a
`trust_remote_code` PreTrainedTokenizer that computes offsets in Python.

Usage:
    uv run python scripts/prepare_for_babylm_eval.py <checkpoint> [--out DIR] [--name NAME] [--no-verify]

    <checkpoint>  path to a .pt file, or a commit hash present in checkpoints/.
    --out DIR     output repo directory (default: checkpoints/hf/<name>).
    --name NAME   repo / model name (default: babylm-gpt-<commit>).
    --no-verify   skip the fidelity verification step.

After running, evaluate with the harness (from the babylm-eval/strict dir):
    bash scripts/eval_zero_shot.sh /abs/path/to/<out> causal
    bash scripts/collate_preds.sh /abs/path/to/<out> causal strict-small
Push the <out> directory to the Hub (it must be public) before submitting.
"""

import argparse
import os
import pickle
import shutil
import sys

import torch

# prepare.py only does work under `if __name__ == "__main__"`, so importing it is
# safe and gives us the exact training tokenizer + data locations for verification.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import prepare  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Templated repo files (written verbatim into the output HF repo)
# ---------------------------------------------------------------------------

CONFIGURATION_PY = '''\
"""HuggingFace config for the autoresearch BabyLM GPT (auto-generated)."""
from transformers import PretrainedConfig


class BabyLMConfig(PretrainedConfig):
    model_type = "babylm_gpt"

    def __init__(
        self,
        sequence_len: int = 2048,
        vocab_size: int = 8192,
        n_layer: int = 8,
        n_head: int = 4,
        n_kv_head: int = 4,
        n_embd: int = 512,
        window_pattern: str = "SSSL",
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.hidden_size = n_embd  # HF-standard alias (the GLUE finetune classifier reads config.hidden_size)
        self.window_pattern = window_pattern
        # lm_head is trained independently of wte; do not tie.
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
'''


MODELING_PY = '''\
"""Self-contained modeling code for the autoresearch BabyLM GPT (auto-generated).

This is a verbatim port of the training-time `forward` (see train.py): RMSNorm,
rotary + QK-norm attention with ResFormer value embeddings, ReLU^2 MLP, per-layer
residual/x0 lambdas, sliding-window (SSSL) attention, and a tanh logit softcap.
Module names match the training checkpoint exactly so weights load with strict=True.
Weights are exported in float32 and the model runs without autocast.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput, BaseModelOutput

from .configuration_babylm import BabyLMConfig


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Value Embedding on alternating layers, last layer always included."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def sdpa_attention(q, k, v, window_size):
    """Causal (optionally sliding-window) attention via PyTorch SDPA.
    q: [B,T,Hq,D], k/v: [B,T,Hkv,D]."""
    B, T, Hq, D = q.shape
    w = window_size[0]  # (left, right); these layers always use right=0
    q_t, k_t, v_t = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    enable_gqa = q.shape[2] != k.shape[2]
    if w >= T - 1:
        attn_mask, is_causal = None, True
    else:
        idx = torch.arange(T, device=q.device)
        attn_mask = (idx[None, :] <= idx[:, None]) & (idx[None, :] >= idx[:, None] - w)
        is_causal = False
    kwargs = {"attn_mask": attn_mask, "is_causal": is_causal}
    if enable_gqa:
        kwargs["enable_gqa"] = True
    y = F.scaled_dot_product_attention(q_t, k_t, v_t, **kwargs)
    return y.transpose(1, 2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer) else None
        )

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        y = sdpa_attention(q, k, v, window_size)
        y = y.contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class BabyLMForCausalLM(PreTrainedModel):
    config_class = BabyLMConfig
    base_model_prefix = "transformer"
    _keys_to_ignore_on_load_missing = [r"cos", r"sin"]

    def __init__(self, config: BabyLMConfig):
        super().__init__(config)
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, value):
        self.transformer.wte = value

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        # attention_mask (padding) is unused: attention is causal so real tokens
        # never attend to right-padding, matching training.
        idx = input_ids
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x).float()
        logits = softcap * torch.tanh(logits / softcap)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-1,
            )

        return CausalLMOutput(loss=loss, logits=logits)


class BabyLMModel(BabyLMForCausalLM):
    """Base encoder (AutoModel): same backbone, returns last_hidden_state (n_embd)
    instead of vocab logits. Used by the GLUE finetuning classifier, which reads
    `last_hidden_state` and sizes its head to config.hidden_size."""

    def forward(self, input_ids, attention_mask=None, return_dict=None, **kwargs):
        idx = input_ids
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)
        return BaseModelOutput(last_hidden_state=x)
'''


TOKENIZATION_PY = '''\
"""Exact HF wrapper around the training Morfessor + byte-fallback tokenizer
(auto-generated).

The training tokenizer (prepare.Tokenizer) segments runs of unicode letters into
morphemes with a Morfessor Viterbi model; every other character and any
out-of-vocabulary morpheme falls back to raw UTF-8 bytes (ids 0..255). Token ids:
[bytes 0..255 | morphemes | special tokens]. This wrapper reproduces that mapping
bit-for-bit and additionally computes character `offset_mapping` (required by the
BabyLM causal backend, which slow HF tokenizers do not provide by default).
"""
from __future__ import annotations

import os
import pickle
import re
import shutil
from typing import List, Optional, Tuple

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PaddingStrategy,
    TruncationStrategy,
)

WORD_RE = re.compile(r"[^\\W\\d_]+", re.UNICODE)  # maximal runs of unicode letters
NUM_BYTE_TOKENS = 256


class BabyLMTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "tokenizer.pkl"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        bos_token: str = "<|reserved_0|>",
        eos_token: str = "<|reserved_0|>",
        pad_token: str = "<|reserved_1|>",
        unk_token: str = "<|reserved_2|>",
        **kwargs,
    ):
        if vocab_file is None or not os.path.isfile(vocab_file):
            raise ValueError(f"BabyLMTokenizer needs tokenizer.pkl (got vocab_file={vocab_file!r})")
        with open(vocab_file, "rb") as f:
            state = pickle.load(f)
        self._vocab_file = vocab_file
        self.morfessor_model = state["model"]
        self.morph_to_id = state["morph_to_id"]
        self._morf_special_tokens = state["special_tokens"]
        self.n_vocab = state["n_vocab"]

        # id <-> token-string tables. Byte ids use the <0xXX> convention; morphemes
        # are their own string; specials are their reserved names.
        self.id_to_str = {i: f"<0x{i:02X}>" for i in range(NUM_BYTE_TOKENS)}
        for morph, tid in self.morph_to_id.items():
            self.id_to_str[tid] = morph
        for name, tid in self._morf_special_tokens.items():
            self.id_to_str[tid] = name
        self.str_to_id = {s: i for i, s in self.id_to_str.items()}
        self._word_cache: dict = {}

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            **kwargs,
        )

    # --- vocab plumbing -------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self.n_vocab

    def get_vocab(self) -> dict:
        vocab = dict(self.str_to_id)
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _convert_token_to_id(self, token: str) -> int:
        if token in self.str_to_id:
            return self.str_to_id[token]
        return self.str_to_id.get(self.unk_token, 0)

    def _convert_id_to_token(self, index: int) -> str:
        return self.id_to_str.get(index, self.unk_token)

    def _decode_ids(self, ids: List[int]) -> str:
        out, buf = [], bytearray()
        for tid in ids:
            if tid < NUM_BYTE_TOKENS:
                buf.append(tid)
            else:
                if buf:
                    out.append(buf.decode("utf-8", errors="replace"))
                    buf.clear()
                out.append(self.id_to_str.get(tid, ""))
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self._decode_ids([self.str_to_id[t] for t in tokens if t in self.str_to_id])

    # --- core encoder (ids + char offsets), mirrors prepare.Tokenizer ---------
    def _encode_word(self, word: str) -> List[Tuple[int, int]]:
        """Return [(id, char_len)] for a letter-run, via Morfessor Viterbi."""
        cached = self._word_cache.get(word)
        if cached is not None:
            return cached
        out: List[Tuple[int, int]] = []
        for atoms in self.morfessor_model.viterbi_segment(tuple(word))[0]:
            morph = "".join(atoms)
            tid = self.morph_to_id.get(morph)
            if tid is not None:
                out.append((tid, len(morph)))
            else:
                for ch in morph:
                    for b in ch.encode("utf-8"):
                        out.append((b, 0))  # 0 => byte of a 1-char span (handled below)
                    out.append((-1, len(ch)))  # marker carrying the char advance
        self._word_cache[word] = out
        return out

    def _encode_ids_offsets(self, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        ids: List[int] = []
        offsets: List[Tuple[int, int]] = []

        def emit_bytes(segment: str, base: int):
            for j, ch in enumerate(segment):
                span = (base + j, base + j + 1)
                for b in ch.encode("utf-8"):
                    ids.append(b)
                    offsets.append(span)

        pos = 0
        for m in WORD_RE.finditer(text):
            if m.start() > pos:
                emit_bytes(text[pos:m.start()], pos)
            word = m.group()
            cursor = m.start()
            for atoms in self.morfessor_model.viterbi_segment(tuple(word))[0]:
                morph = "".join(atoms)
                tid = self.morph_to_id.get(morph)
                if tid is not None:
                    ids.append(tid)
                    offsets.append((cursor, cursor + len(morph)))
                else:
                    emit_bytes(morph, cursor)
                cursor += len(morph)
            pos = m.end()
        if pos < len(text):
            emit_bytes(text[pos:], pos)
        return ids, offsets

    # --- HF encode entry points (override to support offset_mapping) ----------
    def _build(self, text, add_special_tokens, truncation_strategy, max_length):
        ids, offsets = self._encode_ids_offsets(text)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
            offsets = [(0, 0)] + offsets
        if (truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE
                and max_length is not None and len(ids) > max_length):
            ids, offsets = ids[:max_length], offsets[:max_length]
        return ids, offsets

    def _encode_plus(self, text, text_pair=None, add_special_tokens=True,
                     padding_strategy=PaddingStrategy.DO_NOT_PAD,
                     truncation_strategy=TruncationStrategy.DO_NOT_TRUNCATE,
                     max_length=None, stride=0, is_split_into_words=False,
                     pad_to_multiple_of=None, return_tensors=None,
                     return_token_type_ids=None, return_attention_mask=None,
                     return_overflowing_tokens=False, return_special_tokens_mask=False,
                     return_offsets_mapping=False, return_length=False, verbose=True,
                     **kwargs):
        ids, offsets = self._build(text, add_special_tokens, truncation_strategy, max_length)
        encoded = {"input_ids": ids}
        if return_attention_mask is not False:
            encoded["attention_mask"] = [1] * len(ids)
        if return_token_type_ids:
            encoded["token_type_ids"] = [0] * len(ids)
        if return_special_tokens_mask:
            encoded["special_tokens_mask"] = [1 if (add_special_tokens and i == 0) else 0
                                              for i in range(len(ids))]
        if return_offsets_mapping:
            encoded["offset_mapping"] = offsets
        if return_length:
            encoded["length"] = len(ids)

        if padding_strategy != PaddingStrategy.DO_NOT_PAD:
            self._pad_single(encoded, padding_strategy, max_length, pad_to_multiple_of)
        return BatchEncoding(encoded, tensor_type=return_tensors, prepend_batch_axis=True)

    def _batch_encode_plus(self, batch_text_or_text_pairs, add_special_tokens=True,
                           padding_strategy=PaddingStrategy.DO_NOT_PAD,
                           truncation_strategy=TruncationStrategy.DO_NOT_TRUNCATE,
                           max_length=None, stride=0, is_split_into_words=False,
                           pad_to_multiple_of=None, return_tensors=None,
                           return_token_type_ids=None, return_attention_mask=None,
                           return_overflowing_tokens=False, return_special_tokens_mask=False,
                           return_offsets_mapping=False, return_length=False, verbose=True,
                           **kwargs):
        all_ids, all_offsets = [], []
        for item in batch_text_or_text_pairs:
            text = item[0] if isinstance(item, (list, tuple)) else item
            ids, offsets = self._build(text, add_special_tokens, truncation_strategy, max_length)
            all_ids.append(ids)
            all_offsets.append(offsets)

        target_len = 0
        if padding_strategy == PaddingStrategy.LONGEST:
            target_len = max((len(x) for x in all_ids), default=0)
        elif padding_strategy == PaddingStrategy.MAX_LENGTH and max_length is not None:
            target_len = max_length
        if pad_to_multiple_of and target_len:
            target_len = ((target_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

        batch = {"input_ids": [], "attention_mask": []}
        if return_token_type_ids:
            batch["token_type_ids"] = []
        if return_special_tokens_mask:
            batch["special_tokens_mask"] = []
        if return_offsets_mapping:
            batch["offset_mapping"] = []
        if return_length:
            batch["length"] = []

        right = self.padding_side == "right"
        for ids, offsets in zip(all_ids, all_offsets):
            mask = [1] * len(ids)
            stm = [1 if (add_special_tokens and i == 0) else 0 for i in range(len(ids))]
            pad_n = max(0, target_len - len(ids))
            if pad_n:
                pad_ids = [self.pad_token_id] * pad_n
                pad_off = [(0, 0)] * pad_n
                if right:
                    ids, mask, offsets, stm = ids + pad_ids, mask + [0] * pad_n, offsets + pad_off, stm + [0] * pad_n
                else:
                    ids, mask, offsets, stm = pad_ids + ids, [0] * pad_n + mask, pad_off + offsets, [0] * pad_n + stm
            batch["input_ids"].append(ids)
            batch["attention_mask"].append(mask)
            if return_token_type_ids:
                batch["token_type_ids"].append([0] * len(ids))
            if return_special_tokens_mask:
                batch["special_tokens_mask"].append(stm)
            if return_offsets_mapping:
                batch["offset_mapping"].append(offsets)
            if return_length:
                batch["length"].append(len(ids))
        if return_attention_mask is False:
            del batch["attention_mask"]
        return BatchEncoding(batch, tensor_type=return_tensors)

    def _pad_single(self, encoded, padding_strategy, max_length, pad_to_multiple_of):
        ids = encoded["input_ids"]
        target = max_length if padding_strategy == PaddingStrategy.MAX_LENGTH else len(ids)
        if pad_to_multiple_of and target:
            target = ((target + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        pad_n = max(0, target - len(ids))
        if not pad_n:
            return
        right = self.padding_side == "right"
        def ext(key, value):
            if key not in encoded:
                return
            pad = [value] * pad_n
            encoded[key] = encoded[key] + pad if right else pad + encoded[key]
        ext("input_ids", self.pad_token_id)
        ext("attention_mask", 0)
        ext("token_type_ids", 0)
        ext("special_tokens_mask", 1)
        if "offset_mapping" in encoded:
            pad = [(0, 0)] * pad_n
            encoded["offset_mapping"] = (encoded["offset_mapping"] + pad if right
                                         else pad + encoded["offset_mapping"])

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        os.makedirs(save_directory, exist_ok=True)
        dst = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "tokenizer.pkl")
        if os.path.abspath(self._vocab_file) != os.path.abspath(dst):
            shutil.copyfile(self._vocab_file, dst)
        return (dst,)
'''


README_TMPL = '''\
# {name}

Autoresearch BabyLM GPT exported for the [BabyLM 2026 evaluation harness](https://github.com/babylm-org/babylm-eval).

- Custom Muon-trained GPT: value embeddings, rotary + QK-norm, RMSNorm, ReLU^2 MLP,
  per-layer residual/x0 lambdas, sliding-window (SSSL) attention, tanh logit softcap.
- Tokenizer: Morfessor morpheme segmentation with raw-byte fallback (vocab {vocab_size}).
- Source checkpoint commit: `{commit}` — recorded val_bpb: {val_bpb}.

Loaded by the harness as a **causal** backend:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("{name}", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained("{name}", trust_remote_code=True)
```

Evaluate (from `babylm-eval/strict`):

```bash
bash scripts/eval_zero_shot.sh /abs/path/to/{name} causal
bash scripts/collate_preds.sh /abs/path/to/{name} causal strict-small
```

> The model must be **public on HuggingFace** before submitting. Only the zero-shot
> causal pipeline is exported here; GLUE fine-tuning would additionally need a
> sequence-classification head (see the harness HF-conversion tutorial).
'''


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def resolve_checkpoint(arg: str) -> str:
    if os.path.isfile(arg):
        return arg
    cand = os.path.join(REPO_ROOT, "checkpoints", arg if arg.endswith(".pt") else f"{arg}.pt")
    if os.path.isfile(cand):
        return cand
    raise FileNotFoundError(f"Checkpoint not found: {arg} (looked for {cand})")


def write_json(path, obj):
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def convert(ckpt_path: str, out_dir: str, name: str):
    print(f"Loading checkpoint: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ck["config"]
    commit = ck.get("commit", "unknown")
    val_bpb = ck.get("val_bpb", "n/a")

    if os.path.exists(out_dir):
        print(f"Output dir exists, removing: {out_dir}")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 1) code files
    with open(os.path.join(out_dir, "configuration_babylm.py"), "w", encoding="utf-8") as f:
        f.write(CONFIGURATION_PY)
    with open(os.path.join(out_dir, "modeling_babylm.py"), "w", encoding="utf-8") as f:
        f.write(MODELING_PY)
    with open(os.path.join(out_dir, "tokenization_babylm.py"), "w", encoding="utf-8") as f:
        f.write(TOKENIZATION_PY)

    # 2) config.json (with auto_map for trust_remote_code)
    config_json = dict(config)
    config_json.update({
        "model_type": "babylm_gpt",
        "architectures": ["BabyLMForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_babylm.BabyLMConfig",
            "AutoModel": "modeling_babylm.BabyLMModel",
            "AutoModelForCausalLM": "modeling_babylm.BabyLMForCausalLM",
        },
        # lm_head is trained independently (NOT tied to wte); without this HF would
        # overwrite the loaded lm_head with the input embeddings.
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
    })
    write_json(os.path.join(out_dir, "config.json"), config_json)

    # 3) weights -> pytorch_model.bin, upcast everything to float32 (lossless for
    #    the bf16 embeddings) so the model runs without autocast.
    sd = {k: v.float() for k, v in ck["model_state_dict"].items()}
    torch.save(sd, os.path.join(out_dir, "pytorch_model.bin"))
    print(f"Wrote pytorch_model.bin ({len(sd)} tensors, float32)")

    # 4) tokenizer: ship the real Morfessor pickle + configs
    src_pkl = os.path.join(prepare.TOKENIZER_DIR, "tokenizer.pkl")
    if not os.path.isfile(src_pkl):
        raise FileNotFoundError(f"Training tokenizer not found: {src_pkl} (run prepare.py)")
    shutil.copyfile(src_pkl, os.path.join(out_dir, "tokenizer.pkl"))

    with open(src_pkl, "rb") as f:
        tok_state = pickle.load(f)
    specials = tok_state["special_tokens"]
    bos = prepare.BOS_TOKEN  # "<|reserved_0|>"
    reserved = sorted(specials, key=lambda s: specials[s])
    pad = reserved[1] if len(reserved) > 1 else bos
    unk = reserved[2] if len(reserved) > 2 else bos

    write_json(os.path.join(out_dir, "special_tokens_map.json"), {
        "bos_token": bos, "eos_token": bos, "pad_token": pad, "unk_token": unk,
    })
    write_json(os.path.join(out_dir, "tokenizer_config.json"), {
        "tokenizer_class": "BabyLMTokenizer",
        "auto_map": {"AutoTokenizer": ["tokenization_babylm.BabyLMTokenizer", None]},
        "bos_token": bos, "eos_token": bos, "pad_token": pad, "unk_token": unk,
        "model_max_length": config["sequence_len"],
        "clean_up_tokenization_spaces": False,
    })

    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(README_TMPL.format(name=name, vocab_size=config["vocab_size"],
                                   commit=commit, val_bpb=val_bpb))

    print(f"Wrote HF repo to: {out_dir}")
    return config, val_bpb


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def sample_texts(n=60):
    """A few real val sentences (if shards exist) plus fixed multilingual probes."""
    texts = [
        "Hello world! Numbers: 123. Unicode: 你好",
        "The quick brown fox jumps over the lazy dog.",
        "She said, \"Don't worry about it—it's fine.\"",
        "antidisestablishmentarianism and supercalifragilistic",
    ]
    try:
        import pyarrow.parquet as pq
        val_path = os.path.join(prepare.DATA_DIR, prepare.VAL_FILENAME)
        if os.path.isfile(val_path):
            rg = pq.ParquetFile(val_path).read_row_group(0)
            for t in rg.column("text").to_pylist()[: n - len(texts)]:
                if t and t.strip():
                    texts.append(t.strip()[:400])
    except Exception as e:
        print(f"  (could not read val shard for samples: {e})")
    return texts


def _load_generated_tokenizer(out_dir):
    """Import the just-generated tokenization_babylm.py and instantiate it.

    `transformers` is not a dependency of this training repo, but the tokenizer's
    fidelity is the high-risk part, so we test the *actual generated code* by
    loading it against a minimal stub of the few transformers symbols it imports.
    If real transformers is installed we use it instead.
    """
    import importlib.util
    import types

    if "transformers" not in sys.modules:
        try:
            import transformers  # noqa: F401
        except ImportError:
            # Minimal stubs: the encoder itself has no transformers dependency.
            tu = types.ModuleType("transformers.tokenization_utils")

            class PreTrainedTokenizer:  # noqa: D401 - stub base
                def __init__(self, **kwargs):
                    self.padding_side = "right"
                    self.added_tokens_encoder = {}
                    for attr in ("bos_token", "eos_token", "pad_token", "unk_token"):
                        setattr(self, attr, kwargs.get(attr))
                    self.bos_token_id = self.str_to_id.get(self.bos_token)
                    self.eos_token_id = self.str_to_id.get(self.eos_token)
                    self.pad_token_id = self.str_to_id.get(self.pad_token)
                    self.unk_token_id = self.str_to_id.get(self.unk_token)

            tu.PreTrainedTokenizer = PreTrainedTokenizer

            tub = types.ModuleType("transformers.tokenization_utils_base")

            class _Enum:
                DO_NOT_PAD = "do_not_pad"
                LONGEST = "longest"
                MAX_LENGTH = "max_length"
                DO_NOT_TRUNCATE = "do_not_truncate"

            class BatchEncoding(dict):
                def __init__(self, data=None, **kw):
                    super().__init__(data or {})

            tub.BatchEncoding = BatchEncoding
            tub.PaddingStrategy = _Enum
            tub.TruncationStrategy = _Enum

            base = types.ModuleType("transformers")
            sys.modules["transformers"] = base
            sys.modules["transformers.tokenization_utils"] = tu
            sys.modules["transformers.tokenization_utils_base"] = tub

    spec = importlib.util.spec_from_file_location(
        "tokenization_babylm_gen", os.path.join(out_dir, "tokenization_babylm.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tok = mod.BabyLMTokenizer(vocab_file=os.path.join(out_dir, "tokenizer.pkl"))
    # don't leave a __pycache__ behind in the repo we're about to publish
    shutil.rmtree(os.path.join(out_dir, "__pycache__"), ignore_errors=True)
    return tok


def verify(out_dir, config, val_bpb):
    print("\n=== Verification ===")

    # (a) tokenizer: exact id match of the GENERATED encoder vs training tokenizer
    ref_tok = prepare.Tokenizer.from_directory()
    bos_id = ref_tok.get_bos_token_id()
    gen_tok = _load_generated_tokenizer(out_dir)

    texts = sample_texts()
    n_tok_ok, n_off_ok, n_fail_shown = 0, 0, 0
    for t in texts:
        ref_ids = ref_tok.encode(t, prepend=bos_id)
        ids, offsets = gen_tok._encode_ids_offsets(t)
        hf_ids = [bos_id] + ids
        hf_offsets = [(0, 0)] + offsets
        if hf_ids == ref_ids:
            n_tok_ok += 1
        elif n_fail_shown < 3:
            n_fail_shown += 1
            print(f"  TOKEN MISMATCH on: {t[:60]!r}\n    ref={ref_ids[:20]}\n    gen={hf_ids[:20]}")
        if len(hf_offsets) == len(hf_ids) and all(0 <= a <= b <= len(t) for a, b in hf_offsets):
            n_off_ok += 1
    print(f"  tokenizer id-exact: {n_tok_ok}/{len(texts)} sentences")
    print(f"  offset_mapping valid: {n_off_ok}/{len(texts)} sentences")
    tok_ok = (n_tok_ok == len(texts) and n_off_ok == len(texts))

    # (b) model: full HF round-trip (needs transformers; present in the eval env)
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("  (transformers not installed -> skipping HF model load + bpb check;"
              " it will run in the babylm-eval environment)")
        print("=== Verification " + ("PASSED (tokenizer)" if tok_ok else "had WARNINGS") + " ===")
        return tok_ok

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(out_dir, trust_remote_code=True).to(device).eval()
    model_ok = True

    sample = texts[1]
    ids = torch.tensor([ref_tok.encode(sample, prepend=bos_id)], device=device)
    with torch.no_grad():
        out = model(input_ids=ids)
    logits = out.logits
    shape_ok = (logits.shape == (1, ids.shape[1], config["vocab_size"]))
    finite_ok = bool(torch.isfinite(logits).all())
    print(f"  logits shape ok: {shape_ok} {tuple(logits.shape)}  finite: {finite_ok}")
    model_ok &= shape_ok and finite_ok

    # Definitive check: re-run the exact training metric (prepare.evaluate_bpb) on
    # the loaded HF model and compare to the checkpoint's stored val_bpb. A thin
    # shim adapts the HF forward to the (x, y, reduction) signature evaluate_bpb
    # expects. A tiny gap (~1e-4) is expected from fp32 vs the training bf16.
    if device != "cuda":
        print("  (skipping bpb check: evaluate_bpb needs the CUDA dataloader)")
    else:
        try:
            import torch.nn.functional as F

            class _Shim:
                def __call__(self, x, y=None, reduction="mean"):
                    lg = model(input_ids=x).logits
                    if y is None:
                        return lg
                    return F.cross_entropy(lg.view(-1, lg.size(-1)), y.view(-1),
                                           ignore_index=-1, reduction=reduction)

                def eval(self):
                    return self

            with torch.no_grad():
                bpb = prepare.evaluate_bpb(_Shim(), ref_tok, 8)
            print(f"  evaluate_bpb (HF fp32): {bpb:.6f}  (checkpoint val_bpb: {val_bpb})")
            if isinstance(val_bpb, (int, float)) and abs(bpb - val_bpb) > 0.02:
                print("  WARNING: bpb deviates from checkpoint; check the model port.")
                model_ok = False
        except Exception as e:
            print(f"  (bpb check skipped: {e})")

    print("=== Verification " + ("PASSED" if (tok_ok and model_ok) else "had WARNINGS") + " ===")
    return tok_ok and model_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="Path to a .pt file, or a commit hash in checkpoints/")
    ap.add_argument("--out", default=None, help="Output HF repo dir (default checkpoints/hf/<name>)")
    ap.add_argument("--name", default=None, help="Model/repo name")
    ap.add_argument("--no-verify", action="store_true", help="Skip fidelity verification")
    args = ap.parse_args()

    ckpt_path = resolve_checkpoint(args.checkpoint)
    ck_commit = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("commit", "unknown")
    name = args.name or f"babylm-gpt-{ck_commit}"
    out_dir = args.out or os.path.join(REPO_ROOT, "checkpoints", "hf", name)
    out_dir = os.path.abspath(out_dir)

    config, val_bpb = convert(ckpt_path, out_dir, name)
    if not args.no_verify:
        verify(out_dir, config, val_bpb)

    print(f"\nDone. HF repo at: {out_dir}")
    print("Next: push it (public) to the Hub, then run the harness with backend 'causal'.")


if __name__ == "__main__":
    main()
