import os
import json
import torch
import random
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from config import CFG
from dataset import DNABERTDataset
from tokenizer import KmerTokenizer
from model import get_dnabert_model
from torch.optim import AdamW
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_fn(rank, world_size):
    # 分布式设置
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # 设置随机种子
    set_seed(CFG.seed)

    # 加载词表
    with open(CFG.vocab_path, 'r') as f:
        vocab_dict = json.load(f)

    tokenizer = KmerTokenizer(vocab_dict=vocab_dict, k=CFG.k)
    dataset = DNABERTDataset(CFG.train_path, tokenizer, CFG.max_len)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset,
                            batch_size=CFG.batch_size // world_size,
                            sampler=sampler,
                            num_workers=CFG.num_workers)

    model = get_dnabert_model(vocab_dict, CFG)
    model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

    optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    for epoch in range(CFG.epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0

        epoch_iter = tqdm(dataloader, desc=f"[GPU {rank}] Epoch {epoch+1}", disable=(rank != 0))
        for batch in epoch_iter:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            epoch_iter.set_postfix(loss=loss.item())

        if rank == 0:
            print(f"✅ Epoch {epoch+1} Finished | Average Loss: {total_loss/len(dataloader):.4f}")

    # 保存模型（只在主进程）
    if rank == 0:
        os.makedirs(CFG.output_dir, exist_ok=True)
        torch.save(model.module.state_dict(), os.path.join(CFG.output_dir, 'dnabert_ddp.pth'))
        print("✅ 模型已保存（主进程）")

    dist.destroy_process_group()


def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train_fn, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
