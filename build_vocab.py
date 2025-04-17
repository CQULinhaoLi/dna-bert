import json
import itertools
from config import CFG

def generate_kmer_vocab(k):
    vocab = {}
    bases = ['A', 'C', 'G', 'T']
    kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]

    for idx, kmer in enumerate(kmers):
        vocab[kmer] = idx

    # 添加特殊token
    special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
    for token in special_tokens:
        vocab[token] = len(vocab)

    return vocab


if __name__ == '__main__':
    vocab = generate_kmer_vocab(CFG.k)

    save_path = CFG.vocab_path
    with open(save_path, 'w') as f:
        json.dump(vocab, f, indent=4)

    print(f"✅ vocab saved to {save_path}, total size: {len(vocab)}")
