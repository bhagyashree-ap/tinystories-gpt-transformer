# tinystories-gpt-transformer

A decoder only Transformer trained from scratch on TinyStories for next token prediction. The notebook builds a byte level BPE tokenizer, prepares deduplicated train/validation data, trains the model and evaluates generated stories and validation loss.

## Architecture

The pipeline tokenizes TinyStories and splits the text into 256-token input and target sequences. The GPT predicts the next token and cross entropy loss helps AdamW update the model. Performance is measured using validation loss, perplexity and generated text samples.

<img width="861" height="595" alt="tiny-gpt-arch" src="https://github.com/user-attachments/assets/83e15e2b-64ed-48da-9c6d-dbe111d6a8b2" />


### Model

- **Dataset:** TinyStories; 100k training stories and 2000 validation stories.
- **Data preparation:** Normalize story text, deduplicate with SHA-256 hashes, append EOS tokens and construct next token input/target sequences.
- **Tokenizer:** Byte level BPE, trained on training stories only, with an 8,192 token vocabulary.
- **Objective and loss:** Next token prediction with cross entropy loss.
- **Architecture:** Decoder only Transformer with 6 layers and 6 causal self-attention heads.
- **Dimensions:** 384 dimensional embeddings, 256 token context, learned positional embeddings.
- **Transformer block:** Pre-LayerNorm, residual connections, causal multi-head attention and a 4× GELU feed forward network.
- **Regularization:** Dropout 0.1.
- **Output head:** Token embedding weights are tied to the output projection.
- **Sanity checks:** Verify shifted targets, causal masking, cross entropy, finite gradients and checkpoint reload consistency.
- **Fixed batch overfit diagnostic:** Loss decreased from 9.001 to 0.0171 nats/token.

### Training

- **Optimizer:** AdamW, learning rate 3e-4, weight decay 0.1.
- **Learning rate schedule:** 100 update linear warmup followed by cosine decay to 10% of the initial learning rate.
- **Gradient:** Accumulate 16384 tokens per update and clip gradient norm at 1.0.
- **Precision and hardware:** FP16 automatic mixed precision on an NVIDIA A100 40GB.
- **Training budget:** 25 million target tokens, resulting in 1,526 updates (about 25.0M processed tokens).


### Recorded run

- **Validation loss:** 4.257 at step 100, 2.305 at step 1,100, and a best value of 2.221 at step 1,526.
- **Best checkpoint:** Achieved a perplexity of 9.21.
- **Archived results:** [`results/context_256/`](results/context_256/) contains the tokenizer, run manifest, loss history, summary metrics, generated samples and loss curves.
- **Checkpoint comparison:** generation_comparison.md compares the initial, halfway and final checkpoints using three fixed prompts.
- **Generation settings:** Temperature 0.8, top-k 50, maximum of 80 new tokens and EOS stopping.
- **Decoding strategies:** The notebook also compares greedy decoding with temperature 0.7 and temperature 1.0 sampling, both using top-k 50.
