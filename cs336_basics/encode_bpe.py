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

    buffer_size = 4096

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
        buffer: list[int] = []
        with (open(dataset, "r", encoding="utf-8") as input_file,
              open(output_path, "wb") as output_file):
            print(datetime.now(),f"Generating {output_path}")
            for token_id in tokenizer.encode_iterable(input_file):

                buffer.append(token_id)

                if len(buffer) >= buffer_size:
                    np.asarray(buffer, dtype="<u2").tofile(output_file)
                    buffer.clear()

            # 写入不足一个 buffer 的剩余 token
            if buffer:
                np.asarray(buffer, dtype="<u2").tofile(output_file)


if __name__ == "__main__":
    main()