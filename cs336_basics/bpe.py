import os
from typing import BinaryIO
import regex as re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

class BPE():
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
            
    def pre_tokenize_chunk(self,start,end,f,PAT):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        #TODO:support no special token branch
        sub_freq_table = Counter()
        special_tokens_re = [re.escape(special_token) for special_token in self.special_tokens]
        special_tokens_re = "|".join(special_tokens_re)
            
        for sub_chunk in re.splititer(special_tokens_re,chunk):#filter out special token
            for match in re.finditer(PAT,sub_chunk):#pre-token matching
                pre_token = match.group()
                sub_freq_table[tuple(bytes([i]) for i in pre_token.encode("utf-8"))] += 1
        return sub_freq_table
    
    def pre_tokenization(self):

        PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_token_freq_table = Counter()

        with open(self.input_path, "rb") as f:
            num_processes = 4
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = [executor.submit(self.pre_tokenize_chunk,start,end,f,PAT) for start, end in zip(boundaries[:-1], boundaries[1:])]
                for future in futures:
                    pre_token_freq_table.update(future.result())
            
        return pre_token_freq_table
        

    def train(self):  
        self.merge(self.pre_tokenization())

    def merge(self,pre_token_freq_table:dict[tuple[bytes, ...], int]):
        last_freq_table = pre_token_freq_table
        for i in range(self.vocab_size - len(self.vocab)):

            successive_freq_table = Counter()

            for key,val in last_freq_table.items():#find the frequencies of the latest successive pairs
                key_biased = key[1:]
                key_zipped = zip(key,key_biased)
                for pair in key_zipped:
                    successive_freq_table[pair] += val
            if len(successive_freq_table):
                best_pair = max(successive_freq_table,key = lambda pair:(successive_freq_table[pair],pair))
            else:
                break
            new_token = best_pair[0] + best_pair[1]

            next_freq_table = Counter()
            for key,val in last_freq_table.items():#merge the best pair
                new_key = []
                j = 0
                while j < len(key):
                    if (j < len(key) - 1) and (key[j], key[j + 1]) == best_pair:
                        new_key.append(key[j] + key[j + 1])
                        j += 2
                    else:
                        new_key.append(key[j])
                        j += 1
                next_freq_table[tuple(new_key)] += val
            
            self.vocab[len(self.vocab)] = new_token
            self.merges.append(best_pair)

            last_freq_table = next_freq_table



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
