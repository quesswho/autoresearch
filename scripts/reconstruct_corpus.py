"""
Reconstruct a cleaned, deduplicated BabyLM training corpus from the six source
datasets. Pipeline: download (if needed) -> per-source cleaning -> dedup -> write.

The six sources (BabyLM 2026 Strict-Small) are distinct datasets with distinct
noise profiles, so cleaning is per-source:
  childes        - child-directed speech: strip CLAN markup (speaker tags *CHI:,
                   bracket annotations [..], unintelligible xxx/yyy/www, pauses)
  open_subtitles - movie subtitles: de-shout all-caps lines, drop dialogue dashes
                   and music markers
  simple_wiki    - simplified encyclopedia: strip '= = =' section headers, unescape
                   HTML entities (&amp; etc.)
  switchboard    - phone transcripts: strip speaker tags (A:/B:)
  gutenberg      - public-domain books: already clean (HTML-unescape only)
  bnc_spoken     - British spoken: already clean (HTML-unescape only)

Then exact-duplicate documents are removed globally (keep first occurrence). Use
--dedup-min-words to protect short, naturally-frequent utterances (e.g. "yes.").

Output: cleaned <source>.train.txt files in --out, plus a stats report. Point
prepare.py at them (via HOME/cache override) to build shards + train.

Usage:
    uv run python scripts/reconstruct_corpus.py [--out DIR] [--dedup-min-words N]
                                                [--no-download] [--no-dedup]
"""
import os
import re
import sys
import html
import argparse

# prepare.py is import-safe (its work is under __main__); reuse its source list
# and downloader so this script stays in sync with the canonical corpus.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import prepare

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
def n_words(s):
    return len(WORD_RE.findall(s))

# --- per-source cleaning ----------------------------------------------------
BRACKET = re.compile(r"\[[^\]]*\]")               # [leaves room.] CLAN comments/codes
CHILDES_TAG = re.compile(r"^\*[A-Za-z0-9]{1,5}:\s*")
SW_TAG = re.compile(r"^[AB]:\s*")
UNINTELLIGIBLE = re.compile(r"\b(xxx|yyy|www)\b")
PAUSE = re.compile(r"\(\.+\)")                     # (.) (..) pauses
CLAN_SYMS = re.compile(r"\+/+\.|\+\.\.\.|\+!\?|\+\"/\.|[‡„]|\b0\b")
MULTISPACE = re.compile(r"\s{2,}")

def _collapse(s):
    return MULTISPACE.sub(" ", s).strip()

def clean_childes(line):
    if line[:1] in "@%":                           # CHILDES header (@Begin) / tiers (%mor)
        return ""
    line = CHILDES_TAG.sub("", line)
    line = BRACKET.sub(" ", line)
    line = UNINTELLIGIBLE.sub(" ", line)
    line = PAUSE.sub(" ", line)
    line = CLAN_SYMS.sub(" ", line)
    return _collapse(line)

def clean_switchboard(line):
    line = SW_TAG.sub("", line)
    line = UNINTELLIGIBLE.sub(" ", line)
    return _collapse(line)

def clean_subtitles(line):
    line = line.lstrip("- ").strip()               # dialogue dash
    line = line.replace("♪", " ")
    letters = [c for c in line if c.isalpha()]
    if len(letters) >= 3 and sum(c.isupper() for c in letters) / len(letters) > 0.9:
        line = line.lower()                        # de-shout all-caps lines
    return _collapse(line)

def clean_wiki(line):
    line = html.unescape(line)                     # &amp; -> &  etc.
    s = line.strip()
    if s.startswith("=") and s.endswith("="):      # "= = = X = = =" -> "X"
        line = s.strip("= ").strip()
    return _collapse(line)

def clean_identity(line):
    return _collapse(html.unescape(line))

CLEANERS = {
    "childes.train.txt": clean_childes,
    "open_subtitles.train.txt": clean_subtitles,
    "simple_wiki.train.txt": clean_wiki,
    "switchboard.train.txt": clean_switchboard,
    "gutenberg.train.txt": clean_identity,
    "bnc_spoken.train.txt": clean_identity,
}

# --- pipeline ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Reconstruct a cleaned+deduped BabyLM corpus")
    ap.add_argument("--out", default=os.path.join(prepare.CACHE_DIR, "babylm_reconstructed"),
                    help="output dir for cleaned <source>.train.txt files")
    ap.add_argument("--dedup-min-words", type=int, default=4,
                    help="only dedup documents with >= this many words (protects short "
                         "naturally-frequent utterances); 0 = dedup everything")
    ap.add_argument("--target-frac", type=float, default=1.0,
                    help="after clean+dedup, refill each source back to this fraction of "
                         "its ORIGINAL word count by uniformly cycling its unique docs "
                         "(1.0 = original size & distribution at ~10M; 0 = no refill, leaner)")
    ap.add_argument("--no-download", action="store_true", help="skip download (use cached raw)")
    ap.add_argument("--no-dedup", action="store_true", help="skip the dedup step")
    args = ap.parse_args()

    if not args.no_download:
        prepare.download_data()
    os.makedirs(args.out, exist_ok=True)

    seen = set()                  # global exact-dup detection (across all sources)
    print(f"{'source':16} {'orig':>9} {'cleaned':>9} {'deduped':>9} {'refilled':>9} "
          f"{'dup_docs':>8}")
    tot_in = tot_clean = tot_dedup = tot_out = tot_dups = 0
    comp = {}
    for fname in prepare.BABYLM_FILES:
        clean_fn = CLEANERS[fname]
        src = os.path.join(prepare.RAW_DIR, fname)
        dst = os.path.join(args.out, fname)
        wi = wclean = dups = 0
        docs = []                  # unique cleaned (+deduped) docs: (text, n_words)
        with open(src, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                wi += n_words(line)
                c = clean_fn(line)
                if not c or not WORD_RE.search(c):       # dropped empties / pure punctuation
                    continue
                w = n_words(c)
                wclean += w
                if (not args.no_dedup) and w >= args.dedup_min_words:
                    if c in seen:
                        dups += 1
                        continue
                    seen.add(c)
                docs.append((c, w))
        wdedup = sum(w for _, w in docs)
        # Refill to target by uniformly cycling the unique docs (light, even repetition
        # instead of the original's concentrated redundancy).
        target = int(wi * args.target_frac)
        wout = wdedup
        i = 0
        with open(dst, "w", encoding="utf-8") as g:
            for c, _ in docs:
                g.write(c + "\n")
            while wout < target and docs:
                c, w = docs[i % len(docs)]
                g.write(c + "\n")
                wout += w
                i += 1
        print(f"{fname.split('.')[0]:16} {wi:9d} {wclean:9d} {wdedup:9d} {wout:9d} {dups:8d}")
        tot_in += wi; tot_clean += wclean; tot_dedup += wdedup; tot_out += wout; tot_dups += dups
        comp[fname] = wout

    print(f"\nTOTAL words: {tot_in:,} (orig) -> {tot_clean:,} (cleaned) -> {tot_dedup:,} (deduped) "
          f"-> {tot_out:,} (refilled)")
    print(f"  removed by cleaning: {tot_in - tot_clean:,}  |  removed by dedup: {tot_clean - tot_dedup:,} "
          f"({tot_dups:,} duplicate docs)  |  refilled: {tot_out - tot_dedup:,}")
    print(f"  final corpus: {tot_out:,} words  ({'UNDER' if tot_out <= 10_146_225 else 'OVER'} the "
          f"original 10M-budget size)")
    cz = sum(comp.values())
    print("\nfinal composition (word %):")
    for fname in prepare.BABYLM_FILES:
        print(f"  {fname.split('.')[0]:16} {100*comp[fname]/cz:5.1f}%")
    print(f"\nWrote cleaned corpus to: {args.out}")


if __name__ == "__main__":
    main()
