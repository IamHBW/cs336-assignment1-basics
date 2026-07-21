from cs336_basics.bpe import BPE
import argparse
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--vocab_size",type=int,required=True)
    parser.add_argument("--special_tokens",required=True)

    args = parser.parse_args()

    tokenizer = BPE(args.input,args.vocab_size,args.special_tokens)

    tokenizer.train()

    with open("vocab_and_merges.pkl","wb") as f:
        data = {"vocab": tokenizer.vocab,"merges": tokenizer.merges}
        pickle.dump(data,f)



if __name__ == "__main__":
    main()