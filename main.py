
#import all the modules

import torch
import torch.nn as nn
from torch.nn import functional as F

#upload the text file
from google.colab import files
uploaded = files.upload()

#hyperparameters

#batch size is the number of sequences the model will train in parallel before updating the weights
batch_size = 64

#block size is the number of tokens in each independent sequence
block_size = 256

max_iteration = 2000 #the total number of iterations
evaluation_interval = 100 #evaluation of the model in every 100 steps
learning_rate = 3e-4 #0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu' #if the model is running on cpu or gpu
evaluation_iteration = 200 #takes 200 batches to evaluate the loss
embedding_dimension = 64 #hyperparamter
n_head = 4 #attention mechanism running in parallel, so each attention here works with 16d vectors
n_layer = 10 #transformer blocks to get a better token representation and to understand complex texts
dropout = 0.2 #regularization used to reduce overfitting

torch.manual_seed(1337) #used to get the same set of random numbers everytime

#the text file
with open('input2.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#all the unique char occuring in the txt file
chars = sorted(list(set(text)))
print(chars)
vocab_size = len(chars)

#mapping from characters to integers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

#convert string to list of integers
encode = lambda s:[stoi[c] for c in s]

#convert a list of integers to string
decode = lambda l:''.join([itos[i] for i in l])

#train and test splits

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) #traning on 90 percent of the dataset
train_data = data[:n]
validation_data = data[n:]

#data loading

def get_batch(split):
  data = train_data if split == 'train' else validation_data
  index = torch.randint(len(data) - block_size, (batch_size,)) #random starting postions from where sequences will be trained
  x = torch.stack([data[i:i+block_size] for i in index]) # choose a random set of input
  y = torch.stack([data[i+1:i+block_size+1] for i in index]) # choose a random set of target usually starts from the one position right of x
  x, y = x.to(device), y.to(device) #converts to gpu if you are on a cpu
  return x, y

@torch.no_grad() #this is a decorator in pytorch whicch tells to not store or change any value while running the below function
def estimate_loss():
  out = {}
  model.eval() #puts the model in evaluation mode
  for split in ['train', 'validation']: #runs loop twice once during training and during validation
    losses = torch.zeros(evaluation_iteration)#creates a loss tensor
    for k in range(evaluation_iteration):
      X, Y = get_batch(split) #gets a random batch
      logits, loss = model(X, Y) #creates a forward pass, no back propagation is done now (logits are te raw scores of the model before they are converted into probabilties)
      losses[k] = loss.item() #stores the loss
    out[split] = losses.mean() #average of the loss
  model.train() #switches back to training mode after the evaluation is complete
  return out

class Head(nn.Module): #one head of self-attention (inherits the neural network module)

  def __init__(self,head_size):
    super().__init__() #initailize everything the nn.Module needs
    self.key = nn.Linear(embedding_dimension, head_size, bias=False)#converts the token dimension for key
    self.query = nn.Linear(embedding_dimension, head_size, bias=False)#converts the token dimension for query
    self.value = nn.Linear(embedding_dimension, head_size, bias=False)
    self.register_buffer('tril',torch.tril(torch.ones(block_size, block_size))) #this creates a tensor which prevents the model from looking into the future and register tell that this parameter should not be learned
    self.dropout = nn.Dropout(dropout) #dropout layer

  def forward(self, x): #the core where self attention happens
    B, T, C = x.shape #batches, tokens, dimension of the token
    k = self.key(x) #computes key
    q = self.query(x) #computes query
    head_size = k.shape[-1]
    weight = q @ k.transpose(-2, -1) * head_size**-0.5 #computes attention scores, and scales
    weight = weight.masked_fill(self.tril[:T,:T] == 0,float("-inf")) #masking is done
    weight = F.softmax(weight, dim=-1) #converts to probability
    weight = self.dropout(weight) #randomly removes some connections
    v = self.value(x) #compute values
    out = weight @ v #weighted sum
    return out

class MultiHeadAttention(nn.Module): #multiple self attention heads in parallel

  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) #create heads
    self.proj = nn.Linear(embedding_dimension, embedding_dimension) #creates projection which mixes info of all heada
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out

class FeedForward(nn.Module):

  def __init__(self, embedding_dimension):
    super().__init__()
    #layers of functions applied to the info
    self.net = nn.Sequential(
            nn.Linear(embedding_dimension, 4 * embedding_dimension),
            nn.ReLU(),
            nn.Linear(4 * embedding_dimension, embedding_dimension),
            nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module): #attention followed by computation

  def __init__(self, n_embd, n_head):
        # n_head: the number of heads we'd like
        super().__init__()
        head_size = embedding_dimension // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(embedding_dimension)

        #normalization layers
        self.ln1 = nn.LayerNorm(embedding_dimension)
        self.ln2 = nn.LayerNorm(embedding_dimension)

  def forward(self, x):
        #residual connection
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class DecoderOnlyTransformer(nn.Module):

  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, embedding_dimension) #creates a tensor of 64d fro each char
    self.position_embedding_table = nn.Embedding(block_size, embedding_dimension) #positional embedding
    self.blocks = nn.Sequential(*[Block(embedding_dimension, n_head=n_head) for _ in range(n_layer)]) #transformer blocks
    self.ln_final = nn.LayerNorm(embedding_dimension) #final normalization layer
    self.lm_head = nn.Linear(embedding_dimension, vocab_size) #1 extra logit for the next possible character

  def forward(self, index, Targets=None):
    B, T = index.shape
    token_embedding = self.token_embedding_table(index) #creates token embeddings
    position_embedding = self.position_embedding_table(torch.arange(T, device=index.device)) #each position get its own embedding
    x = token_embedding + position_embedding #context + word
    x = self.blocks(x) #pass through all blocks
    x = self.ln_final(x) #final norm
    logits = self.lm_head(x)

    if Targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      Targets = Targets.view(B*T)
      loss = F.cross_entropy(logits, Targets)

    return logits, loss

  def generate(self, index, max_new_tokens): #generation of the tokens
    # index is (B, T) array of indices in the current context
      for _ in range(max_new_tokens):
          # crop index to the last block_size tokens
          index_cond = index[:, -block_size:]
          # get the predictions
          logits, loss = self(index_cond)
          # focus only on the last time step
          logits = logits[:, -1, :] # becomes (B, C)
          # apply softmax to get probabilities
          probs = F.softmax(logits, dim=-1) # (B, C)
          # sample from the distribution
          index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
          # append sampled index to the running sequence
          index = torch.cat((index, index_next), dim=1) # (B, T+1)
      return index

model = DecoderOnlyTransformer()
m = model.to(device)
#total parameters

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

#create pyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

#train the model
for iter in range(max_iteration):

  # every once in a while evaluate the loss on train and val sets
  if iter % evaluation_interval == 0 or iter == max_iteration - 1:
      losses = estimate_loss()
      print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['validation']:.4f}")

  # sample a batch of data
  xb, yb = get_batch('train')

  # evaluate the loss
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True) #clear old gradients
  loss.backward() #backpropagation
  optimizer.step() #optimization (update the weights)

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))

