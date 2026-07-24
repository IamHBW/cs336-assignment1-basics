from cs336_basics.bpe import BPE
import argparse
import pickle
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--vocab_size",type=int,required=True)
    parser.add_argument("--special_tokens",required=True)

    args = parser.parse_args()

    tokenizer = BPE(args.input,args.vocab_size,args.special_tokens)

    tokenizer.train()

    with open(f"vocab_{Path(args.input).stem}.pkl","wb") as f:
        pickle.dump(tokenizer.vocab,f)

    with open(f"merges_{Path(args.input).stem}.pkl","wb") as f:
        pickle.dump(tokenizer.merges,f)


if __name__ == "__main__":
    main()