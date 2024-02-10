import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameter----

load_model = True
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
eval_iters = 200
n_embd = 384
learning_rate = 3e-4
n_heads = 6
n_layer = 6
dropout = 0.2
device = 'cuda.' if torch.cuda.is_available() else 'cpu'

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

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    


## One Head Attention
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # Registering the tril parameters as not a learnable parameter
        #https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723 see this.
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B,T,C(head_size))
        q = self.query(x) #(B, T, C(head_size))

        wei = q @ k.transpose(-2, -1) * C ** -0.5 # (B,T,C) --> (B, C, T) Shape_output: (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)
        # value operation
        v = self.key(x) # (B, T, C(head_size))
        out = wei @ v
        return out
    
class MultiHeadedAttention(nn.Module):
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out
    
class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadedAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x+self.sa(self.ln1(x))
        x = x+self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module): # A simple Bigram model.

    def __init__(self):
        super(BigramLanguageModel, self).__init__() 
        # Creating a embedding table here (a learnable parameter)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) 
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads=n_heads) for _ in range(n_layer)])
        
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets= None): # This function just operate like __call__
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (Batch, Time, Channels) (4, 8(block_size), 32(embd_size)))
        pos_emb = self.position_embedding_table(torch.arange(T,  device = device))
        x = tok_emb + pos_emb # Adding Position Embedidng        

        x = self.blocks(x)
            
        logits = self.lm_head(x) # (B, T, Vocab_size)
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
            
            idx_cond = idx[:, -block_size:] # the attention can be calculated till the block_size

            logits,_ = self(idx_cond)

            logits = logits[:,-1,:] # extracting the last character from the embedding this will of the shape (B, 65). so to predict its next element
            prob = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(prob, num_samples = 1) # (B,1)
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T+1)
        return idx
        

if load_model == True:
    model = torch.load('modelv1.pt')
    model.eval()
    # Generate Text
    idx = torch.zeros((1,1), dtype = torch.long, device = device)
    print(decode(model.generate(idx, max_new_tokens=1000)[0].tolist()))
else:
    model = BigramLanguageModel()
    model.to(device)
    #Optimizer
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"for step: {iter} train loss {losses['train']:.4f}, val_loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = model(xb,yb) #This will calculate loss for every
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    idx = torch.zeros((1,1), dtype = torch.long, device = device)
    print(decode(model.generate(idx, max_new_tokens=1000)[0].tolist()))

