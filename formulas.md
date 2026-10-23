## Model

### • Token + Positional Embeddings
<img width="161" height="61" alt="image" src="https://github.com/user-attachments/assets/f84fc737-df70-44fe-a2ee-61074a5187f5" />

```python
x = self.token(ids) + self.position(pos)[None, :, :]
```
<br>

### • Query, Key, and Value Projections
<img width="249" height="57" alt="image" src="https://github.com/user-attachments/assets/650d6288-ae31-4844-bd21-c284a028dc67" />

```python
q, k, v = self.qkv(x).chunk(3, dim=-1)
```

<br>

### • Scaled Dot-Product Causal Attention

<img width="268" height="167" alt="image" src="https://github.com/user-attachments/assets/fccf7162-0dae-481f-912b-819e4c238c5a" />

```python
scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
scores = scores.masked_fill(mask == 0, float("-inf"))
attn = F.softmax(scores.float(), dim=-1)
out = attn @ v
```

<br>

### • Multi-Head Attention

<img width="212" height="56" alt="image" src="https://github.com/user-attachments/assets/2028f81a-9b1c-4d2a-87d2-a0058d840386" />

```python
# Reshape Q, K, V into six heads
# Compute attention for each head
# Concatenate the heads
out = self.proj(out)
```

<br>

### • Residual Connections + Pre-LayerNorm

<img width="154" height="86" alt="image" src="https://github.com/user-attachments/assets/64dd83c5-1778-4fd9-9168-f7dec805a6fd" />

```python
x = x + self.attn(self.norm1(x))
x = x + self.mlp(self.norm2(x))
```

<br>

### • Feed-Forward Network (GELU)

<img width="269" height="105" alt="image" src="https://github.com/user-attachments/assets/48342c6b-13c2-4c9e-8b71-6287622bf9d4" />

```python
nn.Linear(384, 1536)
nn.GELU()
nn.Linear(1536, 384)
```

<br>

### • Output Projection with Tied Embeddings

<img width="99" height="53" alt="image" src="https://github.com/user-attachments/assets/028d4c3e-6680-4785-8e54-5d0dbf3edf14" />

```python
logits = F.linear(h, self.token.weight)
```

<br>

### • Softmax Probability

<img width="91" height="71" alt="image" src="https://github.com/user-attachments/assets/d2b1a4e0-2c8d-48f8-a5f3-83f7c64fa728" />

```python
F.softmax(scores.float(), dim=-1)
last.softmax(-1)
```

<br>

## Training and Optimization

### • Cross-Entropy Loss

<img width="153" height="75" alt="image" src="https://github.com/user-attachments/assets/9b9a1a06-9f92-416d-bb89-f08d52a82041" />

```python
loss = F.cross_entropy(
    logits.reshape(-1, self.config.vocab_size),
    targets.reshape(-1)
)
```

<br>

### • Linear Learning Rate

<img width="108" height="62" alt="image" src="https://github.com/user-attachments/assets/0387c9fa-7d0b-4dcc-b9be-827a7b78c434" />

```python
lr = c.lr * step / warmup
```

<br>

### • Cosine Learning Rate Decay

<img width="407" height="117" alt="image" src="https://github.com/user-attachments/assets/4a4f5b4e-3e2d-4c1a-aadc-8dfff3c23de7" />

```python
math.cos(math.pi * progress)
```

<br>

### • Gradient Clipping

<img width="149" height="72" alt="image" src="https://github.com/user-attachments/assets/c900660a-b291-4e61-b4d5-aa39b2d2f24a" />
<img width="85" height="20" alt="image" src="https://github.com/user-attachments/assets/f12dfbf4-e78a-4f03-b8cb-5a97b772d01a" />


```python
nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

<br>

## Text Generation and Evaluation

### • Temperature Sampling

<img width="226" height="119" alt="image" src="https://github.com/user-attachments/assets/c97c9929-d659-463f-8643-563dbefc6e62" />

```python
last = last / temperature
probs = last.softmax(-1)
```

<br>

### • Perplexity

<img width="450" height="111" alt="image" src="https://github.com/user-attachments/assets/784ef009-930d-488c-8dcf-d9789e96ebe1" />

```python
"best_val_perplexity": math.exp(best),
```
