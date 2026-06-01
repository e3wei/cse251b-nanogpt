import argparse
import hashlib
import os
import re
from collections import Counter

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default="data/fineweb_score_hq_300M")
parser.add_argument("--train_tokens", type=int, default=300_000_000)
parser.add_argument("--val_tokens", type=int, default=5_000_000)
parser.add_argument("--min_int_score", type=int, default=4)
parser.add_argument("--min_score", type=float, default=3.75)
parser.add_argument("--min_lang_score", type=float, default=0.97)
parser.add_argument("--min_token_count", type=int, default=512)
parser.add_argument("--max_token_count", type=int, default=8192)
parser.add_argument("--max_seen_hashes", type=int, default=2_000_000)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]

space_re = re.compile(r"\s+")
word_re = re.compile(r"[A-Za-z]{2,}")
bad_repeated_char = re.compile(r"(.)\1{25,}")

boilerplate_phrases = [
    "cookie policy",
    "privacy policy",
    "terms of use",
    "all rights reserved",
    "subscribe to our newsletter",
    "enable javascript",
    "javascript is disabled",
    "advertisement",
    "please log in",
    "sign up",
    "share this article",
]

def norm_hash(text):
    text = text.lower()
    text = space_re.sub(" ", text)
    return text.strip()[:20000]

def good_text(text):
    if not text:
        return False
    text = text.strip()
    n = len(text)
    if n < 1500 or n > 120000:
        return False
    if "\x00" in text or bad_repeated_char.search(text):
        return False

    lower = text.lower()
    if sum(p in lower for p in boilerplate_phrases) >= 2:
        return False

    words = word_re.findall(text)
    if len(words) < 220:
        return False

    printable = sum(ch.isprintable() or ch.isspace() for ch in text)
    if printable / max(n, 1) < 0.985:
        return False

    alnum = sum(ch.isalnum() for ch in text)
    if alnum / max(n, 1) < 0.38:
        return False

    weird_symbols = sum(ch in "{}[]<>|\\~^_=*#@$" for ch in text)
    if weird_symbols / max(n, 1) > 0.08:
        return False

    lines = [space_re.sub(" ", ln.strip().lower()) for ln in text.splitlines()]
    lines = [ln for ln in lines if len(ln) >= 20]
    if len(lines) >= 8:
        counts = Counter(lines)
        most_common_frac = counts.most_common(1)[0][1] / len(lines)
        unique_frac = len(counts) / len(lines)
        if most_common_frac > 0.18 or unique_frac < 0.60:
            return False

    return True

ds = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True,
)

train_path = os.path.join(args.out_dir, "train.bin")
val_path = os.path.join(args.out_dir, "val.bin")

seen = set()
train_count = 0
val_count = 0
accepted = 0
rejected_meta = 0
rejected_text = 0
duplicates = 0
target_total = args.train_tokens + args.val_tokens

with open(val_path, "wb") as vf, open(train_path, "wb") as tf:
    pbar = tqdm(total=target_total, unit="tok")

    for doc in ds:
        if doc.get("language") != "en":
            rejected_meta += 1
            continue
        if float(doc.get("language_score", 0.0) or 0.0) < args.min_lang_score:
            rejected_meta += 1
            continue
        if int(doc.get("int_score", 0) or 0) < args.min_int_score:
            rejected_meta += 1
            continue
        if float(doc.get("score", 0.0) or 0.0) < args.min_score:
            rejected_meta += 1
            continue

        tc = int(doc.get("token_count", 0) or 0)
        if tc < args.min_token_count or tc > args.max_token_count:
            rejected_meta += 1
            continue

        text = doc.get("text", "")
        if not good_text(text):
            rejected_text += 1
            continue

        h = hashlib.sha1(norm_hash(text).encode("utf-8", errors="ignore")).hexdigest()
        if h in seen:
            duplicates += 1
            continue
        if len(seen) < args.max_seen_hashes:
            seen.add(h)

        toks = [eot] + enc.encode_ordinary(text)
        arr = np.asarray(toks, dtype=np.uint16)
        accepted += 1

        pos = 0
        while pos < len(arr):
            if val_count < args.val_tokens:
                take = min(len(arr) - pos, args.val_tokens - val_count)
                arr[pos:pos+take].tofile(vf)
                val_count += take
            else:
                take = min(len(arr) - pos, args.train_tokens - train_count)
                arr[pos:pos+take].tofile(tf)
                train_count += take

            pos += take
            pbar.update(take)

            if train_count >= args.train_tokens:
                break

        if train_count >= args.train_tokens:
            break

    pbar.close()

print(f"accepted_docs={accepted:,}")
print(f"rejected_meta={rejected_meta:,}")
print(f"rejected_text={rejected_text:,}")
print(f"duplicates={duplicates:,}")
print(f"wrote {train_count:,} train tokens to {train_path}")
print(f"wrote {val_count:,} val tokens to {val_path}")
