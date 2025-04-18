from Bio import SeqIO  # 用于解析FASTA文件的Biopython模块
import random  # 用于生成随机数
import os  # 用于文件操作

def extract_sequences_from_fasta(fasta_path, output_path, mode='mixed',
                                 min_len=5, max_len=510,
                                 random_count=10000, split_len=510):
    """
    从FASTA文件中提取序列并保存到指定文件中。

    参数:
        fasta_path (str): 输入FASTA文件路径。
        output_path (str): 输出文件路径。
        mode (str): 提取模式，可选 'split', 'random', 'mixed'。
        min_len (int): 随机提取序列的最小长度。
        max_len (int): 随机提取序列的最大长度。
        random_count (int): 随机提取序列的数量。
        split_len (int): 分割模式下每段序列的长度。
    """
    all_sequences = []  # 用于存储提取的所有序列

    # 解析FASTA文件中的每条记录
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq).upper().replace("N", "")  # 转为大写并移除'N'
        seq_len = len(seq)  # 当前序列的长度

        # 如果模式是 'split' 或 'mixed'，按固定长度分割序列
        if mode in ['split', 'mixed']:
            for i in range(0, seq_len - split_len + 1, split_len):
                sub_seq = seq[i:i + split_len]  # 提取子序列
                all_sequences.append(sub_seq)

        # 如果模式是 'random' 或 'mixed'，随机提取子序列
        if mode in ['random', 'mixed']:
            for _ in range(random_count):
                length = random.randint(min_len, max_len)  # 随机生成子序列长度
                if length >= seq_len:  # 如果长度超过原序列，跳过
                    continue
                start = random.randint(0, seq_len - length)  # 随机生成起始位置
                sub_seq = seq[start:start + length]  # 提取子序列
                all_sequences.append(sub_seq)

    # 将提取的序列写入输出文件
    with open(output_path, 'w') as f:
        for seq in all_sequences:
            f.write(seq + '\n')

    # 打印完成信息
    print(f"✅ 提取完成，共生成 {len(all_sequences)} 条序列，保存到 {output_path}")


if __name__ == "__main__":
    # 调用函数提取序列
    extract_sequences_from_fasta(
        fasta_path="./data/ecoli/GCF_000005845.2_ASM584v2_genomic.fna",  # 输入FASTA文件路径
        output_path="./data/ecoli/ecoli_train.txt",  # 输出文件路径
        mode='mixed',  # 提取模式，可选 'split', 'random', 'mixed'
        random_count=30000 # 随机提取序列的数量
    )
