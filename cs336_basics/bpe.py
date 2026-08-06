import os
from typing import BinaryIO
import regex as re
from collections import Counter,defaultdict
from collections.abc import Iterable,Iterator
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from tqdm import tqdm
import pickle

class BPE_train():
    def __init__(self,input_path:str,vocab_size:int,special_tokens:list[str]):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.vocab = {}
        self.merges = []

        for i in range(256):
            self.vocab[i] = bytes([i])
        for i in range(256,256 + len(special_tokens)):
            self.vocab[i] = special_tokens[i - 256].encode("utf-8")
    
    
    def pre_tokenization(self):

        print(datetime.now(),"Beginning pre tokenization")
        pre_token_freq_table = Counter()

        with open(self.input_path, "rb") as f:
            num_processes = 32
            num_chunks = 64
            boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.

            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = [executor.submit(pre_tokenize_chunk,self.input_path,self.special_tokens,start,end) for start, end in zip(boundaries[:-1], boundaries[1:])]
                for future in futures:
                    pre_token_freq_table.update(future.result())
        print(datetime.now(),"Finished pre tokenization")
        return pre_token_freq_table
        

    def train(self):  
        self.merge(self.pre_tokenization())

    def merge(self,pre_token_freq_table:dict[tuple[bytes, ...], int]):
        pair_freq_table = Counter()
        pair_idx_table = defaultdict(set)
        idx = 0
        current_tokens = []
        current_tokens_freq = []
        for key,val in pre_token_freq_table.items():#find the frequencies of the latest successive pairs
                
                key_zipped = list(zip(key,key[1:]))
                current_tokens.append(key)
                current_tokens_freq.append(val)
                for k in range(len(key_zipped)):
                    pair_freq_table[key_zipped[k]] += val
                    pair_idx_table[key_zipped[k]].add(idx)
                
                idx += 1

        for i in tqdm(range(self.vocab_size - len(self.vocab))):
            best_pair = None
            idx_update = set()

            num_pos = sum(value > 0 for value in pair_freq_table.values())
            if num_pos :
                best_pair = max(pair_freq_table,key = lambda pair:(pair_freq_table[pair],pair))
            else:
                break

            new_word = best_pair[0] + best_pair[1]
            for idx in pair_idx_table[best_pair]:
                idx_update.add(idx)

            for idx in idx_update:#update current tokens
                old_key = current_tokens[idx]
                new_key = []
                j = 0
                while j < len(old_key):
                    if (j < len(old_key) - 1) and (old_key[j], old_key[j + 1]) == best_pair:
                        new_key.append(old_key[j] + old_key[j + 1])
                        j += 2
                    else:
                        new_key.append(old_key[j])
                        j += 1

                old_pairs = list(zip(old_key,old_key[1:]))
                new_pairs = list(zip(new_key,new_key[1:]))
                merge_count = defaultdict(int)
                for old_pair in old_pairs:
                    
                    merge_count[old_pair] -= 1
                    pair_idx_table[old_pair].discard(idx)
                
                  
                for new_pair in new_pairs:
                    
                    merge_count[new_pair] += 1

                    pair_idx_table[new_pair].add(idx)

                for pair,count in merge_count.items():
                    pair_freq_table[pair] += current_tokens_freq[idx] * count

                current_tokens[idx] = new_key
                   
                
                                
            
            self.vocab[len(self.vocab)] = new_word
            self.merges.append(best_pair)

class BPE():
    def __init__(self, vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str] | None = None ):
        self.vocab = vocab
        if special_tokens is not None:
            for special_token in special_tokens:
                if special_token.encode("utf-8") not in self.vocab.values():
                    self.vocab[len(vocab)] = special_token.encode("utf-8")
        self.merges = merges
        self.special_tokens = (
            [] if special_tokens is None else special_tokens
            )           
        self.inverse_vocab = {
                word: token_id
                for token_id, word in self.vocab.items()
                }
        self.pair2rank = {
            pair: rank
            for rank,pair in enumerate(self.merges)
        }

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath,"rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath,"rb") as f:
            merges = pickle.load(f)
        return cls(vocab,merges,special_tokens)

    def encode(self, text: str) -> list[int]:
        num_processes = 48
        num_chunks = 48
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            boundaries = find_chunk_boundaries_text(text, num_chunks, sorted_special_tokens[0].encode("utf-8"))
            token_ids = []
            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.

            for start,end in zip(boundaries[:-1], boundaries[1:]):
                token_ids.extend(self.tokenize_chunk_text(text,self.special_tokens,start,end))

            # with ProcessPoolExecutor(max_workers=num_processes) as executor:
            #     futures = [executor.submit(self.tokenize_chunk_text,text,self.special_tokens,start,end) for start, end in zip(boundaries[:-1], boundaries[1:])]
            #     for future in futures:
            #         token_ids.extend(future.result())
        else:
            token_ids = self.tokenize_chunk_text(text,self.special_tokens,0,len(text.encode("utf-8")))

        return token_ids

        


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        text_bytes = b"".join(
            self.vocab[id]
            for id in ids
        )
        return text_bytes.decode("utf-8",errors='replace')

    def tokenize_chunk_text(self,text: str,special_tokens: list[str],start,end):
        
        PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        text_byte = text.encode("utf-8")
        chunk_bytes = text_byte[start:end]
        chunk = chunk_bytes.decode("utf-8")
        sub_token_ids = []
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        #TODO:support no special token branch
        if special_tokens:
            special_token_set = set(special_tokens)
            special_tokens_re = "|".join(
                    re.escape(token)
                    for token in sorted(special_tokens, key=len, reverse=True)
                )
            special_tokens_re = f"({special_tokens_re})"
            for part in re.splititer(special_tokens_re,chunk):
                if part in special_token_set:
                    sub_token_ids.append(self.inverse_vocab[part.encode("utf-8")])
                else:
                    for match in re.finditer(PAT,part):#pre-token matching
                        pre_token = match.group()
                        initial_symbols = [bytes([i]) for i in pre_token.encode("utf-8")]
                        old_symbols = initial_symbols

                        while True:
                            new_symbols = []
                            pair_ranks = []
                            if len(old_symbols) <= 1:
                                break

                            for k in range(0,len(old_symbols) - 1):
                                current_pair = (old_symbols[k], old_symbols[k + 1])
                                if current_pair in self.pair2rank.keys():
                                    pair_ranks.append(self.pair2rank[current_pair])
                                else:
                                    pair_ranks.append(float("inf"))
                            if all(x == float("inf") for x in pair_ranks):
                                break

                            merge_idx = min(range(len(pair_ranks)),key=pair_ranks.__getitem__)

                            j = 0 
                            while j < len(old_symbols):
                                if j == merge_idx:
                                    new_symbols.append(old_symbols[j] + old_symbols[j + 1])
                                    j += 2
                                else:
                                    new_symbols.append(old_symbols[j])
                                    j += 1
                            old_symbols = new_symbols

                        # transform old symbols into ids
                        sub_token_ids.extend([self.inverse_vocab[symbol] for symbol in old_symbols])

        else:
            for match in re.finditer(PAT,chunk):#pre-token matching
                pre_token = match.group()
                initial_symbols = tuple(bytes([i]) for i in pre_token.encode("utf-8"))
                old_symbols = initial_symbols

                while True:
                    new_symbols = []
                    pair_ranks = []
                    if len(old_symbols) <= 1:
                        break

                    for k in range(0,len(old_symbols) - 1):
                        current_pair = (old_symbols[k], old_symbols[k + 1])
                        if current_pair in self.pair2rank.keys():
                            pair_ranks.append(self.pair2rank[current_pair])
                        else:
                            pair_ranks.append(float("inf"))
                    if all(x == float("inf") for x in pair_ranks):
                        break

                    merge_idx = min(range(len(pair_ranks)),key=pair_ranks.__getitem__)

                    j = 0 
                    while j < len(old_symbols):
                        if j == merge_idx:
                            new_symbols.append(old_symbols[j] + old_symbols[j + 1])
                            j += 2
                        else:
                            new_symbols.append(old_symbols[j])
                            j += 1
                    old_symbols = new_symbols
                
                # transform old symbols into ids
                sub_token_ids.extend([self.inverse_vocab[symbol] for symbol in old_symbols])
        
        

        return sub_token_ids


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

def find_chunk_boundaries_text(
    text: str,
    desired_num_chunks: int,
    split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the text into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        text_byte = text.encode("utf-8")
        # Get total text size in bytes
        text_size = len(text_byte)

        chunk_size = text_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = text_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            cursor = initial_position  # Start at boundary guess
            while True:
                mini_chunk = text_byte[cursor:cursor + mini_chunk_size]  # Read a mini chunk
                cursor += mini_chunk_size
                # If EOF, this boundary should be at the end of the text
                if mini_chunk == b"":
                    chunk_boundaries[bi] = text_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

def pre_tokenize_chunk(input_path,special_tokens,start,end):
        
        PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        #TODO:support no special token branch
        sub_freq_table_raw = Counter()
        sub_freq_table = Counter()
        special_tokens_re = [re.escape(special_token) for special_token in special_tokens]
        special_tokens_re = "|".join(special_tokens_re)

        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

        for sub_chunk in re.splititer(special_tokens_re,chunk):#filter out special token
            for match in re.finditer(PAT,sub_chunk):#pre-token matching
                pre_token = match.group()
                sub_freq_table_raw[pre_token] += 1
        for key,val in sub_freq_table_raw.items():
            sub_freq_table[tuple(bytes([i]) for i in key.encode("utf-8"))] = val
        return sub_freq_table

