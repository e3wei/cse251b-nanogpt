import argparse, os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default="data/fineweb_500M")
parser.add_argument("--train_tokens", type=int, default=500_000_000)
parser.add_argument("--val_tokens", type=int, default=5_000_000)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]

ds = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True,
)

train_path = os.path.join(args.out_dir, "train.bin")
val_path = os.path.join(args.out_dir, "val.bin")

train_count = 0
val_count = 0
target_total = args.train_tokens + args.val_tokens

with open(val_path, "wb") as vf, open(train_path, "wb") as tf:
    pbar = tqdm(total=target_total, unit="tok")
    for doc in ds:
        toks = [eot] + enc.encode_ordinary(doc["text"])
        arr = np.asarray(toks, dtype=np.uint16)

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

print(f"wrote {train_count:,} train tokens to {train_path}")
print(f"wrote {val_count:,} val tokens to {val_path}")
