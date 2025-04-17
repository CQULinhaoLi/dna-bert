from Bio import SeqIO
import random
import os

def extract_sequences_from_fasta(fasta_path, output_path, mode='mixed',
                                 min_len=5, max_len=510,
                                 random_count=10000, split_len=510):
    all_sequences = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq).upper().replace("N", "")
        seq_len = len(seq)

        if mode in ['split', 'mixed']:
            for i in range(0, seq_len - split_len + 1, split_len):
                sub_seq = seq[i:i + split_len]
                all_sequences.append(sub_seq)

        if mode in ['random', 'mixed']:
            for _ in range(random_count):
                length = random.randint(min_len, max_len)
                if length >= seq_len:
                    continue
                start = random.randint(0, seq_len - length)
                sub_seq = seq[start:start + length]
                all_sequences.append(sub_seq)

    with open(output_path, 'w') as f:
        for seq in all_sequences:
            f.write(seq + '\n')

    print(f"✅ 提取完成，共生成 {len(all_sequences)} 条序列，保存到 {output_path}")


if __name__ == "__main__":
    extract_sequences_from_fasta(
        fasta_path="./data/ecoli/GCF_000005845.2_ASM584v2_genomic.fna",
        output_path="./data/ecoli/ecoli_train.txt",
        mode='mixed',  # 或 'split'
        random_count=10000
    )
