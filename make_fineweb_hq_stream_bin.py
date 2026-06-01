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
parser.add_argument("--out_dir", default="data/fineweb_hq_500M")
parser.add_argument("--train_tokens", type=int, default=500_000_000)
parser.add_argument("--val_tokens", type=int, default=5_000_000)
parser.add_argument("--max_seen_hashes", type=int, default=3_000_000)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]

boilerplate_phrases = [
    "cookie policy",
    "privacy policy",
    "terms of use",
    "all rights reserved",
    "subscribe to our newsletter",
    "enable javascript",
    "javascript is disabled",
    "advertisement",
    "sign up for our newsletter",
    "please log in",
]

bad_repeated_char = re.compile(r"(.)\1{25,}")
space_re = re.compile(r"\s+")
word_re = re.compile(r"[A-Za-z]{2,}")


def normalize_for_hash(text: str) -> str:
    text = text.lower()
    text = space_re.sub(" ", text)
    return text.strip()[:20_000]


def is_good_doc(text: str) -> bool:
    if not text:
        return False

    text = text.strip()
    n = len(text)

    # Avoid very short junk and absurdly huge pages.
    if n < 800 or n > 200_000:
        return False

    # Hard reject obvious corruption.
    if "\x00" in text or bad_repeated_char.search(text):
        return False

    lower = text.lower()
    boiler_hits = sum(phrase in lower for phrase in boilerplate_phrases)
    if boiler_hits >= 3:
        return False

    # Enough real words.
    words = word_re.findall(text)
    if len(words) < 120:
        return False

    # Avoid symbol/code/table dumps.
    printable = sum(ch.isprintable() or ch.isspace() for ch in text)
    if printable / max(n, 1) < 0.98:
        return False

    alnum = sum(ch.isalnum() for ch in text)
    if alnum / max(n, 1) < 0.30:
        return False

    weird_symbols = sum(ch in "{}[]<>|\\~^_=*#@$" for ch in text)
    if weird_symbols / max(n, 1) > 0.12:
        return False

    # Repeated boilerplate lines.
    lines = [space_re.sub(" ", ln.strip().lower()) for ln in text.splitlines()]
    lines = [ln for ln in lines if len(ln) >= 20]
    if len(lines) >= 8:
        counts = Counter(lines)
        most_common_frac = counts.most_common(1)[0][1] / len(lines)
        unique_frac = len(counts) / len(lines)
        if most_common_frac > 0.20 or unique_frac < 0.55:
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
accepted_docs = 0
rejected_docs = 0
duplicate_docs = 0
target_total = args.train_tokens + args.val_tokens

with open(val_path, "wb") as vf, open(train_path, "wb") as tf:
    pbar = tqdm(total=target_total, unit="tok")

    for doc in ds:
        text = doc.get("text", "")

        if not is_good_doc(text):
            rejected_docs += 1
            continue

        # Exact-ish duplicate removal after normalization.
        h = hashlib.sha1(normalize_for_hash(text).encode("utf-8", errors="ignore")).hexdigest()
        if h in seen:
            duplicate_docs += 1
            continue
        if len(seen) < args.max_seen_hashes:
            seen.add(h)

        toks = [eot] + enc.encode_ordinary(text)
        if len(toks) < 256:
            rejected_docs += 1
            continue

        arr = np.asarray(toks, dtype=np.uint16)
        accepted_docs += 1

        pos = 0
        while pos < len(arr):
            if val_count < args.val_tokens:
                take = min(len(arr) - pos, args.val_tokens - val_count)
                arr[pos:pos + take].tofile(vf)
                val_count += take
            else:
                take = min(len(arr) - pos, args.train_tokens - train_count)
                arr[pos:pos + take].tofile(tf)
                train_count += take

            pos += take
            pbar.update(take)

            if train_count >= args.train_tokens:
                break

        if train_count >= args.train_tokens:
            break

    pbar.close()

print(f"accepted_docs={accepted_docs:,}")
print(f"rejected_docs={rejected_docs:,}")
print(f"duplicate_docs={duplicate_docs:,}")
print(f"wrote {train_count:,} train tokens to {train_path}")
print(f"wrote {val_count:,} val tokens to {val_path}")
