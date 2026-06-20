"""
One-time data preparation for autoresearch experiments.

Downloads the BabyLM 2026 Strict-Small corpus (10M words of developmentally
plausible text), materializes parquet shards (train + a held-out validation
shard), and trains a morphology-based tokenizer (Morfessor).

Usage:
    python prepare.py                  # full prep (download + shards + tokenizer)

Data and tokenizer are stored in ~/.cache/autoresearch/.
"""

import os
import re
import sys
import time
import math
import random
import argparse
import pickle
import collections
from multiprocessing import Pool

import requests
import pyarrow as pa
import pyarrow.parquet as pq
import morfessor
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048       # context length
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
EVAL_TOKENS = 2 * 524288  # ~1.05M tokens for val eval (BabyLM held-out set is small)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
RAW_DIR = os.path.join(CACHE_DIR, "babylm_raw")    # raw BabyLM .txt corpora
DATA_DIR = os.path.join(CACHE_DIR, "data")          # materialized parquet shards
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")

# BabyLM 2026 Strict-Small corpus (10M words of developmentally plausible text).
# https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict-Small
BASE_URL = "https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict-Small/resolve/main"
BABYLM_FILES = [
    "childes.train.txt",
    "gutenberg.train.txt",
    "open_subtitles.train.txt",
    "simple_wiki.train.txt",
    "bnc_spoken.train.txt",
    "switchboard.train.txt",
]

# Shard layout. The raw corpora are plain text (one document per line). We
# deterministically shuffle the documents, hold out a validation slice, and
# write parquet shards with a single "text" column — the format the rest of the
# pipeline (tokenizer training, dataloader, evaluation) already expects.
SHUFFLE_SEED = 1337
VAL_FRAC = 0.1                  # fraction of documents held out for validation
NUM_TRAIN_SHARDS = 8
VAL_SHARD = NUM_TRAIN_SHARDS    # pinned validation shard (highest index)
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"
SHARD_ROW_GROUP_SIZE = 10_000

VOCAB_SIZE = 8192

# Morphology tokenizer config.
# Words (runs of unicode letters) are segmented into morphemes by a Morfessor
# model; everything else (whitespace, punctuation, digits) and any morpheme not
# in the vocabulary falls back to raw UTF-8 bytes, which keeps the tokenizer
# fully lossless. Byte ids occupy 0..255, morpheme ids follow, special tokens
# come last.
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)   # maximal runs of unicode letters
NUM_BYTE_TOKENS = 256                             # raw byte fallback (0..255)
MIN_WORD_COUNT = 2                                # drop hapax words from tokenizer training

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Data download + shard materialization
# ---------------------------------------------------------------------------

def download_single_file(filename):
    """Download one raw BabyLM .txt file with retries. Returns True on success."""
    filepath = os.path.join(RAW_DIR, filename)
    if os.path.exists(filepath):
        return True

    url = f"{BASE_URL}/{filename}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"  Downloaded {filename}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"  Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


def download_data(download_workers=6):
    """Download the raw BabyLM .txt corpora."""
    os.makedirs(RAW_DIR, exist_ok=True)
    existing = sum(1 for f in BABYLM_FILES if os.path.exists(os.path.join(RAW_DIR, f)))
    if existing == len(BABYLM_FILES):
        print(f"Data: all {len(BABYLM_FILES)} raw files already downloaded at {RAW_DIR}")
        return

    needed = len(BABYLM_FILES) - existing
    print(f"Data: downloading {needed} raw files ({existing} already exist)...")

    workers = max(1, min(download_workers, needed))
    with Pool(processes=workers) as pool:
        results = pool.map(download_single_file, BABYLM_FILES)

    ok = sum(1 for r in results if r)
    print(f"Data: {ok}/{len(BABYLM_FILES)} raw files ready at {RAW_DIR}")
    if ok != len(BABYLM_FILES):
        print("Data: some files failed to download — cannot continue.")
        sys.exit(1)


def _read_all_documents():
    """Read every non-empty line across all raw corpora as a list of documents."""
    docs = []
    for filename in BABYLM_FILES:
        filepath = os.path.join(RAW_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(line)
    return docs


def _write_shard(docs, index):
    """Write a list of documents to shard_{index}.parquet with a 'text' column."""
    table = pa.table({"text": docs})
    filepath = os.path.join(DATA_DIR, f"shard_{index:05d}.parquet")
    pq.write_table(table, filepath, row_group_size=SHARD_ROW_GROUP_SIZE)
    return len(docs)


def build_shards():
    """Shuffle documents, hold out a validation slice, write parquet shards."""
    os.makedirs(DATA_DIR, exist_ok=True)
    train_paths = [os.path.join(DATA_DIR, f"shard_{i:05d}.parquet") for i in range(NUM_TRAIN_SHARDS)]
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if all(os.path.exists(p) for p in train_paths) and os.path.exists(val_path):
        print(f"Shards: already materialized at {DATA_DIR}")
        return

    print("Shards: reading raw documents...")
    docs = _read_all_documents()
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(docs)

    n_val = int(len(docs) * VAL_FRAC)
    val_docs = docs[:n_val]
    train_docs = docs[n_val:]
    print(f"Shards: {len(train_docs):,} train docs, {len(val_docs):,} val docs")

    nval = _write_shard(val_docs, VAL_SHARD)
    print(f"  Wrote {VAL_FILENAME} ({nval:,} docs)")

    chunk = math.ceil(len(train_docs) / NUM_TRAIN_SHARDS)
    for i in range(NUM_TRAIN_SHARDS):
        part = train_docs[i * chunk:(i + 1) * chunk]
        n = _write_shard(part, i)
        print(f"  Wrote shard_{i:05d}.parquet ({n:,} docs)")
    print(f"Shards: done, materialized at {DATA_DIR}")

# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def list_parquet_files():
    """Return sorted list of parquet file paths in the data directory."""
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


def text_iterator(max_chars=1_000_000_000, doc_cap=10_000):
    """Yield documents from training split (all shards except pinned val shard)."""
    parquet_paths = [p for p in list_parquet_files() if not p.endswith(VAL_FILENAME)]
    nchars = 0
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def train_tokenizer():
    """Train a Morfessor morphology tokenizer, save vocab + model as a pickle."""
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    parquet_files = list_parquet_files()
    if len(parquet_files) < 2:
        print("Tokenizer: no parquet shards found (need 1 train + 1 val). Run prepare.py first.")
        sys.exit(1)

    # --- Count words across the training split ---
    print("Tokenizer: counting words...")
    t0 = time.time()
    word_counts = collections.Counter()
    for doc in text_iterator():
        word_counts.update(WORD_RE.findall(doc))
    print(f"Tokenizer: {len(word_counts):,} unique words")

    # --- Train Morfessor on the word types ---
    print("Tokenizer: training Morfessor model...")
    model = morfessor.BaselineModel()
    model.load_data([(c, tuple(w)) for w, c in word_counts.items()],
                    freqthreshold=MIN_WORD_COUNT)
    model.train_batch()

    # --- Rank morphemes by corpus frequency and select the vocabulary ---
    # Single-byte (ASCII) morphemes are already covered by the byte fallback,
    # so we don't spend vocab slots on them.
    morph_freq = collections.Counter()
    for word, count in word_counts.items():
        for atoms in model.viterbi_segment(tuple(word))[0]:
            morph_freq["".join(atoms)] += count

    morph_budget = VOCAB_SIZE - NUM_BYTE_TOKENS - len(SPECIAL_TOKENS)
    morphs = []
    for morph, _ in morph_freq.most_common():
        if len(morph.encode("utf-8")) == 1:
            continue
        morphs.append(morph)
        if len(morphs) >= morph_budget:
            break

    # --- Assemble the id space: [bytes | morphs | specials] ---
    morph_to_id = {morph: NUM_BYTE_TOKENS + i for i, morph in enumerate(morphs)}
    specials_offset = NUM_BYTE_TOKENS + len(morphs)
    special_tokens = {name: specials_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    n_vocab = specials_offset + len(SPECIAL_TOKENS)

    state = {
        "morph_to_id": morph_to_id,
        "special_tokens": special_tokens,
        "n_vocab": n_vocab,
        "model": model,
    }
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(state, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    # --- Build token_bytes lookup for BPB evaluation ---
    print("Tokenizer: building token_bytes lookup...")
    token_bytes_list = [1] * NUM_BYTE_TOKENS                    # each byte token is 1 byte
    token_bytes_list += [len(m.encode("utf-8")) for m in morphs]
    token_bytes_list += [0] * len(SPECIAL_TOKENS)              # specials have no byte content
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    # Sanity check
    tok = Tokenizer.from_directory()
    test = "Hello world! Numbers: 123. Unicode: 你好"
    decoded = tok.decode(tok.encode(test))
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={tok.get_vocab_size()})")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Morphology tokenizer. Words are split into morphemes by Morfessor; all
    other text and out-of-vocabulary morphemes fall back to raw UTF-8 bytes, so
    encoding is fully lossless. Training is handled by train_tokenizer above."""

    def __init__(self, state):
        self.model = state["model"]
        self.morph_to_id = state["morph_to_id"]
        self.special_tokens = state["special_tokens"]
        self.n_vocab = state["n_vocab"]
        # id -> string for the non-byte ids (morphemes and specials), used by decode.
        self.id_to_str = {tid: morph for morph, tid in self.morph_to_id.items()}
        self.id_to_str.update({tid: name for name, tid in self.special_tokens.items()})
        self.bos_token_id = self.special_tokens[BOS_TOKEN]
        self._word_cache = {}

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            state = pickle.load(f)
        return cls(state)

    def get_vocab_size(self):
        return self.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def _encode_word(self, word):
        ids = self._word_cache.get(word)
        if ids is not None:
            return ids
        ids = []
        for atoms in self.model.viterbi_segment(tuple(word))[0]:
            morph = "".join(atoms)
            tid = self.morph_to_id.get(morph)
            if tid is not None:
                ids.append(tid)
            else:
                ids.extend(morph.encode("utf-8"))   # byte ids are the byte values 0..255
        self._word_cache[word] = ids
        return ids

    def _encode_ordinary(self, text):
        ids = []
        pos = 0
        for m in WORD_RE.finditer(text):
            if m.start() > pos:
                ids.extend(text[pos:m.start()].encode("utf-8"))
            ids.extend(self._encode_word(m.group()))
            pos = m.end()
        if pos < len(text):
            ids.extend(text[pos:].encode("utf-8"))
        return ids

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.special_tokens[prepend]
        if isinstance(text, str):
            ids = self._encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = [self._encode_ordinary(t) for t in text]
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        out = []
        buf = bytearray()
        for tid in ids:
            if tid < NUM_BYTE_TOKENS:
                buf.append(tid)
            else:
                if buf:
                    out.append(buf.decode("utf-8", errors="replace"))
                    buf.clear()
                out.append(self.id_to_str[tid])
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _document_batches(split, tokenizer_batch_size=128):
    """Infinite iterator over document batches from parquet files."""
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare.py first."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
        assert len(parquet_paths) > 0, "No training shards found."
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    When no document fits remaining space, crops shortest doc to fill exactly.
    100% utilization (no padding).
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    # Pre-allocate buffers: [inputs (B*T) | targets (B*T)]
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    """
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare BabyLM data and tokenizer for autoresearch")
    parser.add_argument("--download-workers", type=int, default=6, help="Number of parallel download workers")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Download raw BabyLM corpora
    download_data(download_workers=args.download_workers)
    print()

    # Step 2: Materialize parquet shards (train + pinned val)
    build_shards()
    print()

    # Step 3: Train tokenizer
    train_tokenizer()
    print()
    print("Done! Ready to train.")
