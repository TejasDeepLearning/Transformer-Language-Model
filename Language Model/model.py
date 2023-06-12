import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataset

import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader

from datetime import datetime

# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         torch.nn.init.xavier_uniform(m.weight)
#         m.bias.data.fill_(0.01)

class CausalSelfAttention(nn.Module):
    def __init__(self, d_k, d_model, n_heads, max_len):
        super().__init__()

        # Assume d_v = d_k
        self.d_k = d_k
        self.n_heads = n_heads

        self.key = nn.Linear(d_model, d_k * n_heads)
        self.query = nn.Linear(d_model, d_k * n_heads)
        self.value = nn.Linear(d_model, d_k * n_heads)

        # final linear layer
        self.fc = nn.Linear(d_k * n_heads, d_model)

        # causal mask
        # make it so that diagonal is 0 too
        # this way we don't have to shift the inputs to make targets
        cm = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer(
            "causal_mask",
            cm.view(1, 1, max_len, max_len)
        )

    def forward(self, q, k, v, pad_mask=None):
        q = self.query(q) # N x T x (hd_k)
        k = self.key(k)   # N x T x (hd_k)
        v = self.value(v) # N x T x (hd_v)

        N = q.shape[0]
        T = q.shape[1]

        # change the shape to:
        # (N, T, h, d_k) -> (N, h, T, d_k)
        # in order for matrix multiply to work properly
        q = q.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(N, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(N, T, self.n_heads, self.d_k).transpose(1, 2)

        # compute attention weights
        # (N, h, T, d_k) x (N, h, d_k, T) --> (N, h, T, T)
        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if pad_mask is not None:
            attn_scores = attn_scores.masked_fill(
                pad_mask[:, None, None, :] == 0, float('-inf'))
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # compute attention-weighted values
        # (N, h, T, T) x (N, h, T, d_k) --> (N, h, T, d_k)
        A = attn_weights @ v

        # reshape it back before final linear layer
        A = A.transpose(1, 2) # (N, T, h, d_k)
        A = A.contiguous().view(N, T, self.d_k * self.n_heads) # (N, T, h*d_k)

        # projection
        return self.fc(A)
    
class TransformerBlock(nn.Module):
    def __init__(self, d_k, d_model, n_heads, max_len, dropout_prob=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = CausalSelfAttention(d_k, d_model, n_heads, max_len)
        self.ann = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout_prob),
        )
        
        # self.ann.apply(init_weights)
        
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x, pad_mask=None):
        x = self.ln1(x + self.mha(x, x, x, pad_mask))
        x = self.ln2(x + self.ann(x))
        x = self.dropout(x)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout_prob=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)

        position = torch.arange(max_len).unsqueeze(1)
        exp_term = torch.arange(0, d_model, 2)
        div_term = torch.exp(exp_term * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x.shape: N x T x D
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
class Decoder(nn.Module):
    def __init__(self,
                vocab_size,
                max_len,
                d_k,
                d_model,
                n_heads,
                n_layers,
                dropout_prob):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
        transformer_blocks = [
            TransformerBlock(
                d_k,
                d_model,
                n_heads,
                max_len,
                dropout_prob) for _ in range(n_layers)]
        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, pad_mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x, pad_mask)
        x = self.ln(x)
        x = self.fc(x) # many-to-many
        return x
    
checkpoint = 'distilbert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_datasets = load_dataset("glue", "sst2")


def tokenize_fn(batch):
  return tokenizer(batch['sentence'], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = tokenized_datasets.remove_columns(
    ["sentence", "idx", "label"])


train_loader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    batch_size=32,
    collate_fn=data_collator
)


model = Decoder(
    vocab_size=tokenizer.vocab_size,
    max_len=tokenizer.max_model_input_sizes[checkpoint],
    d_k=16,
    d_model=768,
    n_heads=12,
    n_layers=2,
    dropout_prob=0.1,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=0)

# A function to encapsulate the training loop
def train(model, criterion, optimizer, train_loader, epochs):
    train_losses = np.zeros(epochs)

    for it in range(epochs):
        model.train()
        t0 = datetime.now()
        train_loss = []
        for batch in train_loader:
            # move data to GPU
            batch = {k: v.to(device) for k, v in batch.items()}

            # zero the parameter gradients
            optimizer.zero_grad()

            # shift targets backwards
            targets = batch['input_ids'].clone().detach()
            targets = torch.roll(targets, shifts=-1, dims=1)
            targets[:, -1] = tokenizer.pad_token_id

            # Forward pass
            outputs = model(batch['input_ids'], batch['attention_mask'])
            # outputs are N x T x V
            # but PyTorch expects N x V x T
            # print("outputs:", outputs)
            # print("targets:", targets)
            loss = criterion(outputs.transpose(2, 1), targets)
            # N, T, V = outputs.shape
            # loss = criterion(outputs.view(N * T, V), targets.view(N * T))
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        scheduler.step()
        
        # Get train loss and test loss
        train_loss = np.mean(train_loss)

        # Save losses
        train_losses[it] = train_loss

        dt = datetime.now() - t0
        print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, Duration: {dt}')

    return train_losses

train_losses = train(
    model, criterion, optimizer, train_loader, epochs=15)

torch.save(model.state_dict(), 
           r'C:\Users\tejas\Documents\Deep Learning\My Work\transformers\gpt\model_state_dict')

valid_loader = DataLoader(
    tokenized_datasets["validation"],
    batch_size=1,
    collate_fn=data_collator
)

model = Decoder(vocab_size=tokenizer.vocab_size,
    max_len=tokenizer.max_model_input_sizes[checkpoint],
    d_k=16,
    d_model=768,
    n_heads=12,
    n_layers=2,
    dropout_prob=0.1)

model.load_state_dict(torch.load(
    r"C:\Users\tejas\Documents\Deep Learning\My Work\transformers\gpt\model_state_dict")
)

model.to('cuda')

model.eval()
for batch in valid_loader:
  # move data to GPU
  batch = {k: v.to(device) for k, v in batch.items()}
  outputs = model(batch['input_ids'], batch['attention_mask'])
  break

# generate something
prompt = "film's"

tokenized_prompt = tokenizer(prompt, return_tensors='pt')

# prepare inputs + get rid of SEP token at the end
input_ids = tokenized_prompt['input_ids'][:, :-1].to(device)
mask = tokenized_prompt['attention_mask'][:, :-1].to(device)

for _ in range(20):
  outputs = model(input_ids, mask)
  prediction_id = torch.argmax(outputs[:, -1, :], axis=-1)

  input_ids = torch.hstack((input_ids, prediction_id.view(1, 1)))
  mask = torch.ones_like(input_ids)

  if prediction_id == tokenizer.sep_token_id:
    break

print(tokenizer.decode(input_ids[0]))
