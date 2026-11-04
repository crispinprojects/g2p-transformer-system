# Transformer-based G2P System

A **Grapheme-to-Phoneme (G2P)** conversion system built from scratch in pure C++17. This project implements a simplified Transformer neural network that learns how English words are pronounced by analysing the **CMU Pronouncing Dictionary**.

The project is split into two core files:

- `g2p_tf_trainer.cpp`: Parses the dictionary, aligns characters to phonemes, trains the network using AdamW + cosine LR scheduling, and serialises the learned weights to a binary file.

- `g2p_tf_tester.cpp`: Loads the trained model and runs **inference** (forward-only prediction) on new words using greedy auto-regressive decoding.

This is a minimal, CPU-optimised educational transformer project to explore how a transformer can be used to solve the G2P problem.

---

## Why Are Transformers So Important?

Transformers revolutionized machine learning by replacing sequential processing (like RNNs/LSTMs) with **parallelizable self-attention**. 

Key reasons why they dominate modern AI:

1. **Long-Range Context**: Every token can directly attend to every other token, capturing dependencies regardless of distance.
2. **Massive Parallelization**: Unlike recurrent networks, transformers process entire sequences simultaneously, enabling efficient GPU/TPU training.
3. **Scalability**: Performance consistently improves with more data and parameters, forming the backbone of modern LLMs, vision models, and multimodal systems.
4. **Unified Architecture**: The same attention-based building blocks work across text, audio, images, and code.

This project implements a **minimal, CPU-optimized educational transformer** to demystify how these components work under the hood.

---

## Transformer Architecture Explained

### What is a Token in G2P?

In this system, a **token** is the smallest unit processed by the network:

- **Input tokens**: Individual characters (graphemes) like `c`, `a`, `t`, plus special markers `<SOS>` (start) and `<EOS>` (end).

- **Output tokens**: Individual phonemes from the CMU dictionary (e.g., `K`, `AE`, `T`), plus `<SOS>`/`<EOS>`.

Tokens are converted to integer IDs, which are then mapped to continuous vectors via **embedding matrices**.

### 1. Input Layer: Embeddings + Positional Encoding

Computers don't understand characters or phonemes directly. The code maps each token ID to a dense vector of size `D_MODEL` (64) using learned embedding matrices (`src_emb`, `tgt_emb`). Since transformers lack inherent sequence order, **sinusoidal positional encodings** are added to each embedding so the network knows the position of each character/phoneme in the sequence.

### 2. Hidden Layer: Transformer Blocks

The core of the network consists of stacked **Transformer Blocks**. Each block contains:

- **Self-Attention / Cross-Attention**: Allows tokens to share information across the sequence.
- **Feed-Forward Network (FFN)**: A two-layer MLP (`Linear → ReLU → Linear`) that processes each token independently after attention.
- **Residual Connections & Layer Normalization**: Stabilize training and allow gradients to flow smoothly through deep networks.

### 3. Output Layer: Logits & Decoding

The final decoder output is projected through a linear layer (`out_proj`) to produce **logits** (raw confidence scores) for every phoneme in the vocabulary. During training, cross-entropy loss compares these logits to the ground-truth phoneme sequence. During inference, the highest-scoring phoneme is selected (greedy decoding).

---

##  Core Concepts: Attention, Q/K/V, and Context

### How Attention Works

Attention is the mechanism that lets tokens "talk" to each other. For every token, the network computes three vectors:

- **Query (Q)**: What this token is *looking for*.
- **Key (K)**: What this token *offers* or represents.
- **Value (V)**: The actual *content* passed when a match is found.

The model computes similarity scores between every Query and every Key (`Q @ K^T`), scales them by `√D_MODEL`, applies softmax to get attention weights, and then computes a weighted sum of the Values. This weighted sum becomes the new context-aware representation of the token.

### Is Attention Sharing Information?

Yes. After the attention step, each token's vector contains a mixture of information from other tokens in the sequence, weighted by relevance. This is how the network resolves ambiguity. For example, the phoneme for `"a"` in `"cat"` vs `"about"` is determined by attending to surrounding characters (`c`, `t` vs `b`, `o`, `u`, `t`).

### Multi-Head Attention vs. This Implementation

Standard transformers use **multi-head attention**, which runs several attention mechanisms in parallel on different subspaces of the embedding, allowing the model to focus on different linguistic features simultaneously. 

 **Note**: This code defines `NUM_HEADS = 4` but currently implements a **unified single-head attention** for simplicity and CPU efficiency. This is for simplification. The `NUM_HEADS` constant is reserved for future expansion.

### What is a "Block" or "Layer"?

A **Transformer Layer** = Self-Attention → Add & Norm → FFN → Add & Norm.  

The **Encoder** stacks these layers. The **Decoder** stacks similar layers but adds **Cross-Attention** (attending to encoder outputs) and **Causal Masking** (preventing future tokens from being seen during training/inference).

### What Does "Deep Learning" Mean Here?

Instead of hand-crafting spelling-to-sound rules, the network automatically learns hierarchical representations of grapheme-phoneme mappings directly from data. The "depth" (`NUM_LAYERS = 2`) allows it to combine local character patterns into broader phonetic rules.

---

##  Configuration Parameters (Mapped to Code)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `D_MODEL` | 64 | Embedding & hidden state dimension |
| `NUM_HEADS` | 4 | Reserved for multi-head expansion (currently single-head) |
| `D_FF` | 128 | Intermediate FFN dimension |
| `NUM_LAYERS` | 2 | Number of encoder/decoder blocks |
| `MAX_SRC_LEN` | 30 | Max input characters per word |
| `MAX_TGT_LEN` | 40 | Max output phonemes per word |
| `BATCH_SIZE` | 8 | Defined but training currently runs sample-by-sample for simplicity |
| `EPOCHS` | 500 | Max training passes over the dataset |
| `LEARNING_RATE` | 1e-3 | Base step size for AdamW |
| `WEIGHT_DECAY` | 0.01 | L2 regularization to prevent overfitting |
| `DROPOUT` | 0.15 | Defined but currently disabled for debugging stability |
| `GRAD_CLIP` | 1.0 | Prevents exploding gradients during backprop |
| `EARLY_STOP_PATIENCE` | 35 | Stops training if validation loss doesn't improve |
| `MAX_TRAINING_WORDS` | 1000 | Limits dataset size for quick CPU testing |

---

## 🔄 Training & Inference Workflow

### Training Phase (`g2p_tf_trainer.cpp`)

1. Loads CMUdict, strips stress markers, and aligns characters to phonemes using linear interpolation.
2. Initializes embeddings, attention weights, FFN matrices, and sinusoidal positional encodings.
3. Runs forward pass → computes cross-entropy loss → backpropagates gradients → updates weights via **AdamW** with cosine learning rate decay.
4. Saves the best validation model to `g2p_tf_model` (raw binary serialization of `std::vector<double>`).

### Inference Phase (`g2p_tf_tester.cpp`)

1. Loads the binary weights into the exact same memory layout.
2. Uses **greedy auto-regressive decoding**: feeds `<SOS>`, gets the most likely phoneme, appends it, repeats until `<EOS>` is predicted.
3. Compares predictions against ground truth using edit distance metrics.

---

## Understanding the Results

### What are WER and CER?

- **WER (Word Error Rate)**: Number of insertions, deletions, or substitutions needed to turn the predicted phoneme sequence into the correct one. Higher = worse.

- **CER (Character/Phoneme Error Rate)**: `WER / max(1, expected_phoneme_count)`. Normalizes the error for word length.

### Initial Results 

Training with 1000 words from CMUdict and setting epochs to 500 gives initial results which look random.

```
cat: FAIL | WER: 23 | CER: 7.67 | Pred: Y G G G B B B B TH TH Y EH K K...
```
This output indicates **severe underfitting**. The model has not learned meaningful patterns yet.

Common reasons in this setup:

1. **Limited Data & Depth**: 1,000 words and only 2 layers are insufficient to capture English phonotactics.
2. **Sample-by-Sample Training**: Processing one word per gradient update introduces high variance. Mini-batching smooths gradients.
3. **Simplified Alignment**: Linear interpolation `(i * phs.size()) / lw.size()` doesn't handle silent letters or variable-length phoneme mappings well.
4. **No Dropout/Regularization**: Disabled for debugging, but can cause memorization or unstable convergence.
5. **Single-Head Attention**: Lacks the parallel feature extraction that multi-head attention provides.

### What to Expect After Proper Training

With more data (~10k+ words), mini-batching, proper alignment (e.g., DTW or CTC), and continued training, the model will start outputting sequences like `K AE T` for `"cat"`. The initial garbage output is normal for a first-pass transformer implementation with low number of words used in training and epochs set to 500.

---

##  Project Structure & Usage

```
g2p_project/
├── trainer/
│   ├── g2p_tf_trainer.cpp   # Doxygen-documented training pipeline
│   ├── Makefile
│   └── cmudict-0.7b
└── tester/
    ├── g2p_tf_tester.cpp    # Doxygen-documented inference engine
    ├── Makefile
    └── g2p_tf_model         # Copied from trainer after training
```

### Compile & Run

```bash
# Install build tools (Debian/Ubuntu)
sudo apt install build-essential

# Train
cd trainer
make
./g2p_tf_trainer cmudict-0.7b

# Test
cd ../tester
cp ../trainer/g2p_tf_model .
make
./g2p_tf_tester
```

---

##  Technical Notes & Code Alignment

- All matrix operations are implemented manually using `std::vector<double>` for zero-dependency C++17 compatibility.
- Layer normalization is applied **post-residual** (as per the original paper).
- Positional encodings use a simplified sinusoidal scheme (`sin` across all dimensions).
- The `Cache` struct stores intermediate activations for efficient backpropagation.
- Weight serialization writes raw `double` arrays to disk, guaranteeing exact reconstruction during inference.

---

## License
Distributed under the **GNU General Public License v3.0**.

## Author
* **Alan Crispin** [Github](https://github.com/crispinprojects)

## 📚 Resources & References

* [CMUdict Corpus](https://github.com/Alexir/CMUdict/blob/master/cmudict-0.7b)
* [CMU Phoneme Symbols](https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b.symbols)
* [Attention Is All You Need](https://arxiv.org/pdf/1706.03762v1)
* [Hugging Face Transformer Course](https://huggingface.co/course)
* [Seq2Seq G2P Survey](https://aclanthology.org/W18-6316/)

---
