import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameter----

batch_size = 32
block_size = 8
max_iters = 500
eval_iters = 20
eval_interval = 50
n_embd = 32
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

#--------------------


with open('input.txt', 'r', encoding = 'utf-8') as f: # Reading the Shakespearian text here
    text = f.read()

vocab = ''.join(sorted(list(set(text))))

vocab_size = len(vocab)

stoi = {ch:i for i,ch in enumerate(vocab)} # This is the dictionary from where we will map from string or in our case character to interger
itos = {i:ch for i,ch in enumerate(vocab)} # this is the dectionary from where we will map from input idx to respective string

encode = lambda s : [stoi[c] for c in s] # This is the encoder
decode = lambda i : ''.join([itos[n] for n in i]) # This is the decoder

# Here we are encoding the whole text into the indexed form based on the dictionary we made or as you can say it as vocab.
data = torch.tensor(encode(text), dtype = torch.long) 

# Dividing the data into train and validation set
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)

def get_batch(split = 'train'):
    data = train_data if split == 'train' else val_data
    # Here we split the data len(data) - block_size as if the randomized form the end no so it has sufficient amount of buffer to load the data.
    ix = torch.randint(len(data) - block_size, (batch_size,)) 
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # This is setting up the model for evaluation
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module): # A simple Bigram model.

    def __init__(self):
        super(BigramLanguageModel, self).__init__() 
        # Creating a embedding table here (a learnable parameter)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets= None): # This function just operate like __call__
        tok_emb = self.token_embedding_table(idx) # (Batch, Time, Channels) (4, 8(block_size), 65(vocab_size))
        logits = self.lm_head(tok_emb) #(B, T, Vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens): # With the help of this function we will predict the next character
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:,-1,:] # extracting the last character from the embedding this will of the shape (B, 65). so to predict its next element
            prob = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(prob, num_samples = 1) # (B,1)
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T+1)
        return idx
        

model = BigramLanguageModel()
m = model.to(device)


# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"for step: {iter} train loss {losses['train']:.4f}, val_loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = m(xb,yb) #This will calculate loss for every
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Genrate Text
idx = torch.zeros((1,1), dtype = torch.long, device = device)
print(decode(m.generate(idx, max_new_tokens=200)[0].tolist()))

