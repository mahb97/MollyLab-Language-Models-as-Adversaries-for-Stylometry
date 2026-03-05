# Architectural Specification: Custom Transformer Components

## Overview

MollyLab's generation and evaluation pipeline is built around two custom transformer components, each trained from scratch on the Joycean corpus. Neither is a pretrained model with adapters. Both are designed specifically for the task of stylometric style modelling across Joyce's four compositional tiers.

$$\mu p \rightarrow \text{UP}$$

The formula maps directly onto the two-component architecture:

| Symbol | Component | Role |
|---|---|---|
| **μ** | Morpheme-aware vocabulary pipeline | The atomic units — Joycean coinages treated as irreducible |
| **→** | Component A: Generative Transformer | The transformation — decoding stylistically coherent prose from those units |
| **UP** | Component B: Style Encoder | The emergent geometry — a learned style manifold where position means something |

Component A is the attacker. Component B is the space the attacker learns to navigate. P3's direction consistency loss is the mechanism that connects them: it enforces that Component A's generated trajectories move coherently through the style manifold that Component B has learned.

---

## Motivation for Training from Scratch

The standard approach — PEFT-adapting a pretrained LLM — is retained in MollyLab as the primary generator suite (G-T1 through G-full). The custom transformer components constitute a **second experimental track** that addresses a fundamental limitation of the PEFT approach.

Pretrained LLMs are trained on internet-scale text where literary prose is a small minority. Their embedding geometry reflects the statistical structure of the web. When PEFT-adapted on Joyce, the model is being redirected against billions of parameters worth of priors that encode nothing about literary style. The adaptation is swimming against a very strong current.

A transformer trained from scratch on the Joyce corpus has an embedding space whose geometry is **entirely shaped by the stylistic structure of the training data**. Every dimension of every embedding vector is a response to Joycean text and nothing else. This has two consequences:

First, the generative transformer's internal representations are stylometrically meaningful in a way that pretrained models' representations are not. There is no internet-English substrate to fight against.

Second, and more importantly: when Component B (the Style Encoder) is trained on the same vocabulary and the same corpus, the style manifold it learns and the representational space that Component A inhabits are **commensurate**. P3's trajectory loss operates in a space that the generator actually lives in, rather than using a pretrained model's hidden states as an approximate proxy.

This commensurateness is the architectural contribution. It is not achievable with pretrained models.

**The honest limitation**: the Joyce corpus is approximately 1.05 million words, or 5–6 million tokens — small by transformer standards. Training from scratch on this data requires careful regularisation and scale discipline. The architecture is deliberately small; scaling beyond the specified parameter counts risks memorisation rather than generalisation, which would invalidate the adversarial evaluation.

---

## The Vocabulary Pipeline

The vocabulary pipeline is the foundational layer that both components share. It is where μp → UP begins.

### Stage 1 — Joycean Coinage Registry

Before any tokenisation, a curated registry of Joycean atomic units is compiled. These are lexical items that standard BPE would fragment incorrectly because it has no basis for treating them as units:

**Category 1 — Wake portmanteau neologisms**: compound coinages where the meaning is carried by the fusion, not the components. `funferall`, `chaosmos`, `cropse`, `bababadalgharaghtakamminarronnkonnbronntonnerronntuonnthunntrovarrhounawnskawntoohoohoordenenthurnuk` (the thunderword), and the full catalogue of Wakean portmanteaux. Scholars have catalogued several thousand of these; the registry draws on existing computational Joyce scholarship and manual curation.

**Category 2 — Ulysses coinages and compounds**: Joyce's non-standard compounds, dialect forms, and stream-of-consciousness fusions in the middle and late Ulysses episodes. Less extensively catalogued than Wake neologisms but present throughout T3 and concentrated in T3-P (Penelope).

**Category 3 — Cross-tier protected forms**: proper nouns with Joycean orthographic specificity, recurring non-standard spellings, and dialect phonetic transcriptions that appear across multiple tiers and should be consistently treated as atomic.

The registry is versioned and documented in `data/vocabulary/registry.json`. It is a research output in its own right.

### Stage 2 — Protected Pre-tokenisation

Before BPE training runs, every token in the registry is replaced with a unique protected placeholder string that BPE cannot split (format: `<JOYCE_UNIT_XXXX>` where XXXX is the registry index). BPE then runs on the placeholder-substituted corpus. After BPE training, placeholders are restored as single vocabulary entries.

The result: a vocabulary where `funferall` and `chaosmos` are **atomic vocabulary items** alongside standard BPE subwords. They are μ — irreducible units that carry meaning at the morpheme level.

### Stage 3 — BPE Training

BPE is trained on the full Joyce corpus (all four tiers concatenated) with the protected pre-tokenisation applied. Vocabulary size: **16,000 tokens**. This is deliberately small — large vocabularies on small corpora produce sparse embeddings that generalise poorly. At 16k, the vocabulary is rich enough to capture Joycean specificity while maintaining sufficient token frequency for stable embedding learning.

The final vocabulary comprises:
- Protected Joycean atomic units (registry entries)
- BPE subword units learned from the remaining corpus
- Standard special tokens (`<BOS>`, `<EOS>`, `<PAD>`, `<MASK>`, `<UNK>`)
- Tier-marker tokens (`<T1>`, `<T2>`, `<T3>`, `<T3P>`, `<T4>`) — used during Component B training

**This vocabulary is shared by both Component A and Component B.** This is non-negotiable for the commensurateness argument: the style encoder must be measuring distance in the same token space that the generative transformer is generating in.

---

## Component A — The Generative Transformer

### Architecture

A **decoder-only transformer** following the standard autoregressive language modelling architecture, scaled for the Joyce corpus.

| Parameter | Value | Rationale |
|---|---|---|
| Layers | 6 | Sufficient depth for stylistic structure; deeper risks memorisation on 6M tokens |
| Attention heads | 8 | Standard for embedding dimension 512 |
| Embedding dimension | 512 | Balances representational capacity against corpus size |
| FFN dimension | 2048 | 4× embedding dimension, standard ratio |
| Vocabulary size | 16,000 | Shared morpheme-aware vocabulary |
| Context window | 512 tokens | Sufficient for paragraph-level stylistic coherence across all tiers |
| Parameters (approx.) | ~40M | Calculated below |
| Positional encoding | Rotary (RoPE) | Better length generalisation than learned absolute positions; handles Penelope's long unpunctuated spans |
| Attention | Causal self-attention with FlashAttention-2 | Standard autoregressive masking |
| Normalisation | Pre-LayerNorm (before attention and FFN) | More stable training than post-LN on small corpora |
| Activation | SwiGLU | Standard modern choice; better gradient flow than ReLU |

**Parameter count breakdown:**
- Token embedding matrix: 16,000 × 512 = ~8.2M
- 6 × attention layers (Q, K, V, O projections): 6 × 4 × 512 × 512 = ~6.3M
- 6 × FFN layers (two linear + gating): 6 × 3 × 512 × 2048 = ~18.9M
- 6 × LayerNorm pairs: negligible
- LM head (tied with embedding): 0 additional (weight tying)
- **Total: ~33–40M parameters**

Weight tying between the input embedding matrix and the LM head output projection is enforced. This is standard practice and is particularly important here: it means the model's output distribution is directly constrained by the same embedding geometry that μp → UP describes. The output logits are inner products with embedding vectors — the model is literally projecting back into μ-space to decide what to say next.

### Training Regime

Component A is trained in three phases corresponding to μp → UP:

**P1 — Embedding warm-up (μ)**
Freeze all layers except the token embedding matrix. Train on the target tier corpus with a high learning rate (1e-3) for 500–1000 steps. This teaches the model the vocabulary geometry of the target tier before any syntactic or compositional learning begins. The protected registry tokens receive gradient signal that positions them correctly relative to their compositional context in the tier.

**P2 — Full model training (→)**
Unfreeze all layers. Train with standard autoregressive cross-entropy loss and a cosine learning rate schedule (peak 3e-4, warmup 500 steps). Train until validation loss plateaus, with early stopping patience of 5 evaluations. This is the transformation phase — the model learns to route morpheme-level units through the attention machinery to produce stylistically coherent continuations.

**P3 — Direction consistency enforcement (UP)**
Apply $\mathcal{L}_{P3}$ as defined in `docs/p3_loss.md`, using Component B's encoder (trained separately, see below) to compute $\bar{e}_T$ and the embedding trajectory signal $\mathcal{L}_e$. Run for a fixed number of steps (tuned per tier) with low learning rate (1e-5). This enforces that the trained model's generations move coherently through the style manifold across extended sequences.

**One model per tier, plus one full-corpus model:**

| Model | Training data | Notes |
|---|---|---|
| **JoyceGen-T1** | *Dubliners* | Smallest training set; aggressive regularisation (dropout 0.2) |
| **JoyceGen-T2** | *Portrait* | |
| **JoyceGen-T3** | *Ulysses* (full) | |
| **JoyceGen-T3P** | *Ulysses*, Penelope only | Smallest training set after T1; context window reduced to 256 for training |
| **JoyceGen-T4** | *Finnegans Wake* | Largest training set; dropout reduced to 0.1 |
| **JoyceGen-full** | All tiers concatenated | Tier-marker tokens prepended to each document during training |

### Relationship to PEFT Generator Suite

The JoyceGen models are a **parallel experimental track**, not a replacement for the PEFT generators (G-T1 through G-full). Both tracks are evaluated in the attack matrices. The comparison between them is itself a finding: does a from-scratch model trained on 40M parameters and the target corpus attack stylometric classifiers more or less effectively than a 1B PEFT-adapted model? The answer is not obvious and the result is publishable either way.

---

## Component B — The Style Encoder

### Architecture

An **encoder-only transformer** trained to map text chunks into a learned style manifold via a hierarchical classification objective.

| Parameter | Value | Rationale |
|---|---|---|
| Layers | 6 | Matched to Component A for comparability |
| Attention heads | 8 | Standard |
| Embedding dimension | 256 | Smaller than Component A — classification requires less capacity than generation |
| FFN dimension | 1024 | 4× embedding dimension |
| Vocabulary size | 16,000 | Shared with Component A |
| Context window | 512 tokens | Matched to Component A |
| Parameters (approx.) | ~15–18M | |
| Positional encoding | Learned absolute | Encoder; absolute positions are sufficient and slightly simpler |
| Attention | Bidirectional self-attention | Full context; no causal masking |
| Normalisation | Pre-LayerNorm | |
| Activation | GELU | Standard for encoder architectures |

### Hierarchical Training Objective

Component B is trained with a **two-level hierarchical classification objective** that encodes two distinct geometric axes into the style manifold:

**Level 1 — Joyce manifold boundary (binary)**
A binary classifier head trained to distinguish Joyce text from non-Joyce prose. This axis encodes the boundary of the Joyce manifold — what makes Joyce *Joyce* in general, independent of period. The non-Joyce class uses the control corpus (same corpus used as negative class in the attack–defence evaluation).

**Level 2 — Evolutionary position (4-way)**
A 4-way classifier head trained to predict tier membership (T1, T2, T3, T4) on Joyce text only. This axis encodes position along the evolutionary arc within the Joyce manifold.

The two heads share the transformer encoder backbone and are trained jointly:

$$\mathcal{L}_{B} = \mathcal{L}_{\text{binary}} + \lambda_h \cdot \mathcal{L}_{\text{tier}}$$

where $\mathcal{L}_{\text{binary}}$ is binary cross-entropy (Joyce vs. non-Joyce) and $\mathcal{L}_{\text{tier}}$ is categorical cross-entropy over the four tiers, applied only to Joyce samples. $\lambda_h$ is a balancing weight (default 0.5; tuned on validation set).

T3-P (Penelope) is **not** a separate class in the 4-way objective — it is included within T3. It is tracked as a sub-class for analysis but not given a separate classification target, which would create a severe class imbalance (24k words vs. 265k words for the rest of T3).

### Training

Component B is trained on the full Joyce corpus plus the non-Joyce control corpus, with document-level splits matched to the evaluation splits used in the attack–defence evaluation. It is trained **before** P3 of Component A begins, because its encoder output is used as the embedding trajectory signal $e_i$ in $\mathcal{L}_{P3}$.

The classification heads are discarded after training. The encoder backbone is frozen and used as a **fixed style embedding function** throughout P3 and evaluation. It is never updated after its initial training.

**Why freeze it**: if Component B's encoder were updated during Component A's P3 training, the target style manifold would shift under the generator's feet. The centroids $\bar{e}_T$ must represent human-authored source text, not the model's current approximation of it.

### The Style Manifold

After training, the encoder's `[CLS]` token representation (or mean-pooled final hidden state) maps any text chunk to a 256-dimensional style vector. This vector space is the **style manifold**.

The hierarchical objective predicts that this space will have interpretable geometric structure:

- **Axis 1 (Joyce boundary)**: Joyce text should be separable from non-Joyce text by a hyperplane learned at Level 1
- **Axis 2 (evolutionary arc)**: within the Joyce subspace, T1 through T4 should be ordered along a trajectory that corresponds to the chronological stylistic evolution — not necessarily linear, but with T1 and T4 at opposite extremes

This geometric prediction is **falsifiable** and is tested explicitly in `analysis/style_manifold.py` using UMAP visualisation and linear probing of the encoder's output space.

**The Woolf experiment** places Woolf text into this manifold without any retraining. The question becomes geometric: does Woolf land inside the Joyce manifold or outside it? At what position along the evolutionary axis? Does Penelope-style Woolf (Mrs Dalloway interior monologue) land closer to T3-P than to the non-Joyce region? These are answerable questions with a well-defined style space.

---

## How the Two Components Connect

The connection is P3, but it is worth stating it precisely.

After Component B is trained and frozen, its encoder $\text{Enc}_B$ serves as the embedding function for P3's trajectory loss. The centroid $\bar{e}_T$ used in $\mathcal{L}_e$ is:

$$\bar{e}_T = \frac{1}{|D_T|} \sum_{d \in D_T} \text{Enc}_B(d)$$

computed over all documents $d$ in the target tier's training split, using the frozen Component B encoder.

During P3 training of Component A, the embedding $e_i$ of each generated window $w_i$ is:

$$e_i = \text{Enc}_B(w_i)$$

also computed with the frozen Component B encoder.

This means **P3 is enforcing that Component A's generated text moves toward the target tier's centroid in Component B's style manifold** — a space organised around Joyce-specific stylometric structure, using a vocabulary that treats Joycean coinages as atomic units.

The full pipeline from vocabulary to generation to evaluation:

```
Joycean Coinage Registry
        ↓
Morpheme-aware BPE Vocabulary (shared)
        ↓                    ↓
Component A              Component B
Generative Transformer   Style Encoder
P1: embedding warm-up    Level 1: Joyce boundary
P2: full LM training     Level 2: tier arc
P3: trajectory loss  ←── frozen encoder output
        ↓
Generated text
        ↓
Attack–Defence Evaluation Matrix
(alongside PEFT generator suite)
```

---

## Feasibility and Training Schedule

### Before September (current hardware)

Assuming a single consumer GPU (RTX 3090 / 4090 class, 24GB VRAM):

| Task | Estimated time | Notes |
|---|---|---|
| Registry curation (T4) | 2–3 weeks | Manual + computational; largest effort |
| Registry curation (T1–T3) | 1 week | Smaller; T3-P can reuse T3 work |
| Vocabulary pipeline implementation | 3–5 days | BPE training on 6M tokens is fast |
| Component B training (full corpus) | 4–6 hours | Small encoder; fast convergence |
| Component A P1+P2, one tier (prototype) | 12–24 hours | JoyceGen-T3P recommended as first prototype |
| Component A P3, one tier | 4–8 hours | Depends on P3 step count |

**Recommended pre-September milestone**: working prototype of JoyceGen-T3P (Penelope only) with Component B trained, P3 running, and one full attack run against D1-A-T3 classifier. This validates the full pipeline end-to-end before HPC scale-up.

### September onwards (HPC)

| Task | Estimated time (A100 80GB) |
|---|---|
| Full vocabulary pipeline | 1–2 days |
| Component B training | 2–4 hours |
| JoyceGen-T1 through T4, all phases | 2–3 days total |
| JoyceGen-full | 1 day |
| Full attack matrix (JoyceGen suite) | 3–5 days |
| Comparative analysis vs. PEFT suite | 1–2 days |

---

## Implementation Notes

### File locations

```
mollylab/
│
├── data/
│   └── vocabulary/
│       ├── registry.json          # Joycean coinage registry (versioned)
│       ├── build_vocab.py         # Protected pre-tokenisation + BPE training
│       └── tokeniser/             # Trained tokeniser artefacts
│
├── models/
│   ├── component_a/
│   │   ├── config.py              # Architecture hyperparameters
│   │   ├── model.py               # Decoder-only transformer implementation
│   │   └── train.py               # P1/P2/P3 training loop
│   └── component_b/
│       ├── config.py
│       ├── model.py               # Encoder + hierarchical classification heads
│       ├── train.py               # Hierarchical training loop
│       └── encode.py              # Frozen encoder inference (used by P3 and analysis)
│
└── analysis/
    └── style_manifold.py          # UMAP visualisation, linear probing, Woolf placement
```

### Key implementation constraints

**Weight tying in Component A**: the LM head must be explicitly tied to the input embedding matrix. Do not allow them to diverge during training.

**Component B must be fully trained and frozen before Component A P3 begins**: the training scripts enforce this dependency. `train.py` for Component A P3 will raise an error if no frozen Component B checkpoint is found.

**Registry is immutable after vocabulary training begins**: any changes to `registry.json` after BPE training invalidate the trained vocabulary and require retraining both components from scratch. Version the registry with git tags.

**Shared vocabulary path**: both `models/component_a/config.py` and `models/component_b/config.py` point to the same tokeniser artefact in `data/vocabulary/tokeniser/`. Never copy the tokeniser — always reference the shared path.

---

## Relationship to the Broader MollyLab Architecture

The custom transformer components are a **second experimental track** within MollyLab, running in parallel with the PEFT generator suite. They are not a replacement for it. The full experimental design includes both:

| Track | Generators | Primary contribution |
|---|---|---|
| **PEFT track** | G-T1, G-T2, G-T3, G-T3P, G-T4, G-full | Adversarial evaluation of existing stylometry methods under realistic threat models |
| **Custom transformer track** | JoyceGen-T1 through JoyceGen-full | Architectural contribution: from-scratch style modelling with morpheme-aware vocabulary and learned style manifold |

The comparison between tracks is itself a research finding. If JoyceGen models attack classifiers more effectively than PEFT models at matched parameter scales, it supports the commensurateness argument — that stylometrically coherent generation benefits from training in a purpose-built representational space. If not, it tells you something about the relative importance of scale versus architectural specificity for stylometric attack strength.

Either result is publishable. Both results together are a dissertation chapter.

---

## Connection to Wake2vec

The morpheme-aware vocabulary pipeline and the three-phase P1/P2/P3 training regime are conceptually descended from Wake2vec, where the same formula — μp → UP — anchors the architecture. The implementations are distinct:

| | Wake2vec | MollyLab Component A |
|---|---|---|
| **Corpus** | *Finnegans Wake* only | Full Joycean arc (T1–T4) |
| **μ-units** | Wake neologisms and portmanteaux | Joycean coinages across all tiers |
| **P3 signal** | Morpheme group composition coherence | Joint embedding + stylometric trajectory in Component B's style manifold |
| **Companion architecture** | None | Component B (Style Encoder) |
| **Objective** | How embeddings bend under extreme style | Whether the bent embeddings break classifiers |

Wake2vec asks what happens when you push a model *into* a voice. MollyLab asks whether what comes *out* can survive adversarial scrutiny. The formula travels. The architecture does not copy.

---

*From the smallest unit of meaning, the whole voice rises.*

$$\mu p \rightarrow \text{UP}$$
