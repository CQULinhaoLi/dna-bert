import os
import json
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW

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

from tqdm import tqdm  # 新增

# ===== 2. 训练函数 =====
def train_fn(model, dataloader, optimizer):
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
    model.to(CFG.device)  # ✅ 必须先移动模型到 device

    # 多卡并行
    if torch.cuda.device_count() > 1:
        print(f"🔧 使用 {torch.cuda.device_count()} 块GPU并行训练")
        model = torch.nn.DataParallel(model)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    # 训练
    for epoch in range(CFG.epochs):
        print(f"\n--- Epoch {epoch + 1}/{CFG.epochs} ---")
        avg_loss = train_fn(model, train_loader, optimizer)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

    # 保存模型
    os.makedirs(CFG.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CFG.output_dir, 'dnabert_pretrained.pth'))
    print("✅ 模型已保存。")


if __name__ == '__main__':
    main()
    print("🚀 训练完成！")