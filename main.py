import os
import json
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


from config import CFG
from dataset import DNABERTDataset
from tokenizer import KmerTokenizer  # 你自己定义的分词器类
from model import get_dnabert_model  # 你自己定义的模型获取函数


# ===== 1. 设置随机种子 =====
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

from tqdm import tqdm  

# ===== 2. 训练函数 =====
def train_fn(model, dataloader, optimizer, scheduler):
    model.train()
    total_loss = 0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training', leave=True)

    for step, batch in progress_bar:
        input_ids = batch['input_ids'].to(CFG.device)
        attention_mask = batch['attention_mask'].to(CFG.device)
        labels = batch['labels'].to(CFG.device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)

        loss = outputs.loss
        loss = loss.mean()
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        progress_bar.set_postfix({'step': step + 1, 'loss': f"{loss.item():.4f}"})

    return total_loss / len(dataloader)

# ===== 3. 主函数 =====
def main():
    # set seed
    set_seed(CFG.seed)

    # load vocab
    with open(CFG.vocab_path, 'r') as f:
        vocab_dict = json.load(f)

    # initialize tokenizer
    tokenizer = KmerTokenizer(vocab_dict=vocab_dict, k=CFG.k)

    # Dataset & Dataloader
    train_dataset = DNABERTDataset(CFG.train_path, tokenizer, CFG.max_len)
    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers)

    # 模型
    model = get_dnabert_model(vocab_dict, CFG)
    model.to(CFG.device)

    # 多卡并行
    is_parallel = False
    if torch.cuda.device_count() > 1:
        print(f"🔧 使用 {torch.cuda.device_count()} 块GPU并行训练")
        model = torch.nn.DataParallel(model)
        is_parallel = True

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    total_steps = len(train_loader) * CFG.epochs
    warmup_steps = int(CFG.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    # 训练
    for epoch in range(CFG.epochs):
        print(f"\n--- Epoch {epoch + 1}/{CFG.epochs} ---")
        avg_loss = train_fn(model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

    # ===========================
    # ✅ 模型保存（结构 + 权重 + tokenizer + optimizer）
    # ===========================
    os.makedirs(CFG.output_dir, exist_ok=True)

    # 保存模型结构和权重
    if is_parallel:
        model.module.save_pretrained(CFG.output_dir)
    else:
        model.save_pretrained(CFG.output_dir)

    # 保存 tokenizer（你自定义的 KmerTokenizer，这里只保存 vocab）
    with open(os.path.join(CFG.output_dir, f"vocab{CFG.k}.json"), "w") as f:
        json.dump(tokenizer.vocab, f, indent=4)
    with open(os.path.join(CFG.output_dir, "tokenizer_config.json"), "w") as f:
        json.dump({"k": CFG.k}, f, indent=4)

    # 保存 optimizer 和 scheduler（可选）
    torch.save({
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, os.path.join(CFG.output_dir, "optimizer_scheduler.pt"))

    # 保存训练参数
    train_args = {
        # 基本设置
        "project": CFG.project,
        "seed": CFG.seed,

        # 数据设置
        "k": CFG.k,
        "max_len": CFG.max_len,
        "train_path": CFG.train_path,
        "vocab_path": CFG.vocab_path,

        # 模型设置
        "hidden_size": CFG.hidden_size,
        "num_hidden_layers": CFG.num_hidden_layers,
        "num_attention_heads": CFG.num_attention_heads,
        "intermediate_size": CFG.intermediate_size,
        "type_vocab_size": CFG.type_vocab_size,

        # 训练设置
        "batch_size": CFG.batch_size,
        "num_workers": CFG.num_workers,
        "epochs": CFG.epochs,
        "lr": CFG.lr,
        "weight_decay": CFG.weight_decay,
        "warmup_ratio": CFG.warmup_ratio,
        "max_grad_norm": CFG.max_grad_norm,

        # 保存设置
        "output_dir": CFG.output_dir,
        "save_steps": CFG.save_steps,
        "logging_steps": CFG.logging_steps,

        # 其他
        "use_fp16": CFG.use_fp16
    }

    with open(os.path.join(CFG.output_dir, "train_args.json"), "w") as f:
        json.dump(train_args, f, indent=4)

    print(f"✅ 模型与相关信息已保存到：{CFG.output_dir}")



if __name__ == '__main__':
    main()
    print("🚀 训练完成！")