"""Reconstruct a cleaned BabyLM corpus: strip transcription/markup noise while
preserving the actual language and the source mix. Result is < 10M words.
Reads original raw files, writes cleaned ones to babylm_clean/."""
import os, re, html, collections

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
SRC = os.path.join(CACHE, "babylm_raw")
DST = os.path.join(CACHE, "babylm_clean")
os.makedirs(DST, exist_ok=True)
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
def wc(s): return len(WORD_RE.findall(s))

BRACKET = re.compile(r"\[[^\]]*\]")            # [leaves room.] CLAN comments/codes
CHILDES_TAG = re.compile(r"^\*[A-Za-z0-9]{1,5}:\s*")
SW_TAG = re.compile(r"^[AB]:\s*")
UNINTELLIGIBLE = re.compile(r"\b(xxx|yyy|www)\b")
PAUSE = re.compile(r"\(\.+\)")                  # (.) (..) pauses
CLAN_SYMS = re.compile(r"\+/+\.|\+\.\.\.|\+!\?|\+\"/\.|[‡„]|\b0\b")
WIKI_HEAD = re.compile(r"^=+\s*(.*?)\s*=+\s*$")
MULTISPACE = re.compile(r"\s{2,}")

def collapse(s):
    return MULTISPACE.sub(" ", s).strip()

def clean_childes(line):
    if line[:1] in "@%":            # CHILDES header (@Begin) / dependent tiers (%mor)
        return ""
    line = CHILDES_TAG.sub("", line)
    line = BRACKET.sub(" ", line)
    line = UNINTELLIGIBLE.sub(" ", line)
    line = PAUSE.sub(" ", line)
    line = CLAN_SYMS.sub(" ", line)
    return collapse(line)

def clean_switchboard(line):
    line = SW_TAG.sub("", line)
    line = UNINTELLIGIBLE.sub(" ", line)
    return collapse(line)

def clean_subtitles(line):
    line = line.lstrip("- ").strip()            # dialogue dash
    line = line.replace("♪", " ")
    letters = [c for c in line if c.isalpha()]
    if len(letters) >= 3 and sum(c.isupper() for c in letters)/len(letters) > 0.9:
        line = line.lower()                     # de-shout all-caps lines
    return collapse(line)

def clean_wiki(line):
    line = html.unescape(line)                  # &amp; -> &  etc.
    s = line.strip()
    if s.startswith("=") and s.endswith("="):   # "= = = X = = =" -> "X"
        line = s.strip("= ").strip()
    return collapse(line)

def identity(line):
    return collapse(html.unescape(line))

CLEANERS = {
    "childes": clean_childes, "switchboard": clean_switchboard,
    "open_subtitles": clean_subtitles, "simple_wiki": clean_wiki,
    "gutenberg": identity, "bnc_spoken": identity,
}

print(f"{'source':16} {'words_in':>9} {'clean':>9} {'kept%':>6} {'refilled':>9} {'dup%':>6}")
tot_in = tot_clean = tot_out = 0
comp = {}
for src, fn in CLEANERS.items():
    inp = os.path.join(SRC, src+".train.txt")
    out = os.path.join(DST, src+".train.txt")
    docs=[]; wi=wclean=0
    with open(inp, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            wi += wc(line)
            c = fn(line)
            if not c or not WORD_RE.search(c):   # drop empties / pure-punct leftovers
                continue
            w = wc(c); wclean += w
            docs.append((c, w))
    # Refill back to the original per-source word count by cycling cleaned docs
    # (same source distribution, full budget, but the old noise-budget is now
    # clean repeated content instead of transcription markup).
    wo = wclean; i = 0; refilled = 0
    with open(out, "w", encoding="utf-8") as g:
        for c,_ in docs:
            g.write(c+"\n")
        while wo < wi and docs:
            c, w = docs[i % len(docs)]
            g.write(c+"\n"); wo += w; refilled += 1; i += 1
    dup = 100*refilled/max(len(docs),1)
    print(f"{src:16} {wi:9d} {wclean:9d} {100*wclean/max(wi,1):6.1f} {wo:9d} {dup:6.1f}")
    tot_in += wi; tot_clean += wclean; tot_out += wo; comp[src] = wo

print(f"\nTOTAL words: orig {tot_in:,} -> cleaned {tot_clean:,} -> refilled {tot_out:,}  (noise replaced by clean dup: {tot_in-tot_clean:,} words)")
print("\nComposition (word %):  source: orig -> clean")
# recompute orig composition
orig = {}
for src in CLEANERS:
    orig[src] = sum(wc(l) for l in open(os.path.join(SRC,src+'.train.txt'),encoding='utf-8'))
oz=sum(orig.values()); cz=sum(comp.values())
for src in CLEANERS:
    print(f"  {src:16} {100*orig[src]/oz:5.1f}% -> {100*comp[src]/cz:5.1f}%")
