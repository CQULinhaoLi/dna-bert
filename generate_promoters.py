import random
import os

def generate_sequence(length=200):
    return ''.join(random.choices('ACGT', k=length))

def insert_promoter(seq, promoter="TATAAT", position=50):
    return seq[:position] + promoter + seq[position+len(promoter):]

def generate_dataset(n_samples=100, promoter_ratio=0.5, output_file="toy_promoter_data.tsv"):
    with open(output_file, 'w') as f:
        f.write("sequence\tlabel\n")
        for _ in range(n_samples):
            seq = generate_sequence()
            if random.random() < promoter_ratio:
                seq = insert_promoter(seq)
                label = 1
            else:
                label = 0
            f.write(f"{seq}\t{label}\n")

os.makedirs('data/promoter/toy/',exist_ok=True)
generate_dataset(n_samples=100, output_file="data/promoter/toy/toy_promoter_train.tsv")
generate_dataset(n_samples=20, output_file="data/promoter/toy/toy_promoter_dev.tsv")
generate_dataset(n_samples=20, output_file="data/promoter/toy/toy_promoter_test.tsv")

print("OK")
