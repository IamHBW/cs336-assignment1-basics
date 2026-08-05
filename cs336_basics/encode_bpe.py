import os
from typing import BinaryIO
from cs336_basics.bpe import BPE,find_chunk_boundaries
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

def main():
    owt_paths = ["data/owt_train.txt","data/owt_valid.txt"]
    TinyStories_paths = ["data/TinyStoriesV2-GPT4-train.txt","data/TinyStoriesV2-GPT4-valid.txt"]
    output_dir = Path("outputs/ids")
    output_dir.mkdir(parents=True, exist_ok=True)


    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--special_tokens",nargs="+",default=["<|endoftext|>"])

    args = parser.parse_args()
    assert args.dataset in ("owt","stories")

    if args.dataset == "owt":
        tokenizer = BPE.from_files("outputs/tokenizers/vocab_owt_train.pkl","outputs/tokenizers/merges_owt_train.pkl",args.special_tokens)
        dataset_paths = owt_paths
    else:
        tokenizer = BPE.from_files("outputs/tokenizers/vocab_TinyStoriesV2-GPT4-train.pkl","outputs/tokenizers/merges_TinyStoriesV2-GPT4-train.pkl",args.special_tokens)
        dataset_paths = TinyStories_paths

    num_chunks = 64
    for dataset in dataset_paths :
        output_path = output_dir / f"{Path(dataset).stem}.npy"
        with (open(dataset, "rb") as input_file,
              open(output_path, "wb") as output_file):
            print(datetime.now(),f"Generating {output_path}")
            boundaries = find_chunk_boundaries(input_file,num_chunks,b"<|endoftext|>")
            i = 0
            for start,end in zip(boundaries[:-1], boundaries[1:]):
                token_ids = encode_chunk(dataset,start,end,tokenizer)
                np.asarray(token_ids, dtype="<u2").tofile(output_file)
                print(datetime.now(),f"Written {output_path} block {i}")
                i += 1

def encode_chunk(
    input_path: BinaryIO,
    start: int,
    end: int,
    tokenizer: BPE
    ) -> list[int]:
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
        return tokenizer.encode(text)


if __name__ == "__main__":
    main()