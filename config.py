import torch
class CFG:
    # ========= 基本设置 =========
    project = 'DNABERT_pretrain'
    seed = 42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ========= 数据设置 =========
    k = 6  # k-mer大小
    max_len = 512  # 最长token数（包含[CLS]和[SEP]）
    train_path = 'data/ecoli/ecoli_train.txt'  # 原始DNA序列
    vocab_path = f'data/ecoli/kmer_vocab_k{k}.json'    

    # ========= 模型设置 =========
    vocab_size = 4 ** k + 5  # 4^6 + 5（特殊token），可以运行后自动计算
    hidden_size = 768
    num_hidden_layers = 12
    num_attention_heads = 12
    intermediate_size = 3072
    type_vocab_size = 1  # 没有NSP任务，只用一种segment

    # ========= 训练设置 =========
    batch_size = 1024
    num_workers = 8 * 4
    epochs = 10
    lr = 4e-4
    weight_decay = 0.01
    warmup_ratio = 0.1
    max_grad_norm = 1.0
    

    # ========= 模型保存 =========
    output_dir = './outputs/ecoli_pretrain'
    save_steps = 500
    logging_steps = 50

    # ========= 其他 =========
    use_fp16 = False  # 如果你用AMP混合精度训练
