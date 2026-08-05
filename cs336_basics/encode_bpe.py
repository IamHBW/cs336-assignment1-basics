from cs336_basics.bpe import BPE
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


    for dataset in dataset_paths :
        output_path = output_dir / f"{Path(dataset).stem}.npy"
        print(datetime.now(),f"Generating {output_path}")
        with open(dataset, "r", encoding="utf-8") as f:
            text = f.read()
            token_ids =  np.asarray(tokenizer.encode(text),dtype=np.uint16)
            np.save(output_path,token_ids)


if __name__ == "__main__":
    main()