# P3: Direction Consistency Loss

## Overview

MollyLab's generation pipeline is structured in three phases, each corresponding to one term of the guiding formula:

$$\mu p \rightarrow \text{UP}$$

drawn from Joyce's *Ulysses* (Circe episode, Episode 15), where the colon in `up:UP` reads simultaneously as a mapping operator, a type annotation, and a definition. From the smallest unit of meaning, the whole voice rises.

| Phase | Symbol | Operation |
|---|---|---|
| **P1** | μ | Embedding adaptation — the model learns the target tier's micro-units |
| **P2** | → | LoRA attention routing — the transformation through the model's machinery |
| **P3** | UP | Direction consistency — the transformation is enforced as a proper morphism, not a lookup table |

P3 is the subject of this document. It defines a joint loss over two trajectories — embedding space and stylometric space — that enforces coherent directional movement toward the target tier's voice across a generated sequence, rather than merely local feature matching at the token level.

P3 cannot run before P1 and P2 are complete. There is nothing to enforce trajectory consistency *over* until the μ-units exist and the routing machinery is in place.

---

## Motivation

A generator that passes P1 and P2 evaluation may still exhibit two failure modes that P3 is designed to correct:

**Register lurching**: the generated sequence oscillates between the target tier's stylistic register and the base model's generic register, window by window. Local features look correct; global voice coherence is absent. A stylometric classifier operating over the full sequence will detect the inconsistency even if individual windows pass.

**Centroid proximity without trajectory**: the generated text sits near the target tier's centroid in embedding space on average, but the movement across the sequence is random rather than directed. The generator has learned what the target region of representation space looks like but not how to *stay* there.

P3 addresses both failure modes through a joint penalty over sequential windows of generated text: an embedding trajectory component that enforces directional consistency in representation space, and a stylometric trajectory component that enforces register smoothness across the surface features the attribution classifiers will actually see.

---

## Formal Definition

### Notation

Let a generated sequence be divided into $n$ sliding windows $w_1, w_2, \ldots, w_n$ of fixed token length $L$ with stride $s < L$ (overlapping windows).

For each window $w_i$, define:

- $e_i \in \mathbb{R}^d$ — mean-pooled embedding of $w_i$ over the model's final hidden layer
- $f_i \in \mathbb{R}^k$ — stylometric feature vector of $w_i$ (see §Feature Vector below)
- $\bar{e}_T \in \mathbb{R}^d$ — centroid of target tier $T$ in embedding space, **precomputed and frozen** over the training split before P3 begins
- $\bar{f}_T \in \mathbb{R}^k$ — centroid of target tier $T$ in stylometric feature space, **precomputed and frozen** over the training split before P3 begins

### Embedding Trajectory Component $\mathcal{L}_e$

$$\mathcal{L}_e = \frac{1}{n}\sum_{i=1}^{n} \left(1 - \cos(e_i,\ \bar{e}_T)\right) + \lambda_e \sum_{i=2}^{n} \max\left(0,\ \cos(e_{i-1},\ \bar{e}_T) - \cos(e_i,\ \bar{e}_T)\right)$$

**Term 1** — centroid pull: draws each window's embedding toward the target tier centroid. Operates independently per window.

**Term 2** — directional consistency penalty: a one-sided hinge that penalises any window whose embedding is *further* from the target centroid than the preceding window's. It does not reward getting closer; it only penalises getting further away.

The one-sided formulation is deliberate. A two-sided penalty would force monotonic convergence toward the centroid, which over-constrains generation and would pathologically penalise the natural variation present even within a single tier. The hinge allows free movement toward or parallel to the target; it only fires when the sequence is actively retreating.

### Stylometric Trajectory Component $\mathcal{L}_s$

$$\mathcal{L}_s = \frac{1}{n}\sum_{i=1}^{n} \left\|f_i - \bar{f}_T\right\|_2 + \lambda_s \sum_{i=2}^{n} \left\|f_i - f_{i-1}\right\|_2$$

**Term 1** — centroid pull in stylometric space: draws each window's feature vector toward the target tier's stylometric centroid.

**Term 2** — register smoothness penalty: a two-sided $\ell_2$ penalty on the difference between adjacent windows' feature vectors. Unlike the embedding component, this is symmetric — any sharp stylometric discontinuity between adjacent windows is penalised regardless of direction.

The asymmetry between $\mathcal{L}_e$ (one-sided) and $\mathcal{L}_s$ (two-sided) reflects the different nature of each space. Embedding trajectories have a meaningful direction (toward or away from the centroid); stylometric lurching in either direction is equally a failure of register coherence.

### Joint Loss with Curriculum Schedule

$$\mathcal{L}_{P3} = \alpha(t) \cdot \mathcal{L}_e + \beta(t) \cdot \mathcal{L}_s$$

where $t$ is the training step within P3 and the weights follow a crossover curriculum:

$$\alpha(t) = \alpha_0 \cdot e^{-\gamma t}$$

$$\beta(t) = \beta_0 \cdot \left(1 - e^{-\gamma t}\right)$$

**Early training** ($t \approx 0$): $\alpha \approx \alpha_0$, $\beta \approx 0$. The embedding trajectory loss dominates — the model learns to navigate toward the correct region of representation space. Stylometric consistency cannot be enforced before the model is pointing in the right direction.

**Late training** ($t \gg 0$): $\alpha \approx 0$, $\beta \approx \beta_0$. The stylometric trajectory loss dominates — the model learns to maintain register coherence at the surface level, which is what the attribution classifiers evaluate.

The crossover rate $\gamma$ is set empirically, calibrated to when the embedding loss plateaus within P3. In practice, monitor $\mathcal{L}_e$ independently; when its rate of improvement drops below a threshold $\epsilon_e$, increase $\gamma$ to accelerate the handoff to $\mathcal{L}_s$.

---

## Stylometric Feature Vector $f_i$

The feature vector $f_i$ is computed from the **surface text** of window $w_i$ only — not from the model's internal representations. This is a deliberate design constraint.

**Rationale**: the adversarial evaluation in MollyLab tests whether generated text fools stylometric classifiers that operate on surface features. If $f_i$ used the model's internal representations as a proxy, P3 would be optimising against a signal unavailable to the actual classifiers during evaluation. Surface-only computation ensures that what P3 enforces is exactly what the classifiers measure.

$f_i$ comprises the following components, concatenated into a single vector:

| Component | Dimension | Description |
|---|---|---|
| Function-word frequencies | 200 | Relative frequency of top-200 English function words |
| Character 3-gram distribution | 500 | Relative frequencies of most common character trigrams in training split |
| Character 4-gram distribution | 500 | Relative frequencies of most common character 4-grams in training split |
| Punctuation density | 6 | Commas, full stops, em-dashes, ellipses, semicolons, colons per 100 tokens |
| Mean sentence length | 1 | Mean tokens per sentence in the window |
| Sentence length std | 1 | Standard deviation of sentence lengths |
| MTLD (lexical diversity) | 1 | Measure of Textual Lexical Diversity over the window |

Total: $k = 1{,}209$ dimensions. The n-gram components are computed relative to the frequency distribution of the **training split only**, frozen before P3 begins.

---

## Window Parameters

Window length $L$ and stride $s$ should be tuned per tier, reflecting the natural scale of stylistic variation in each corpus:

| Tier | Recommended $L$ | Recommended $s$ | Rationale |
|---|---|---|---|
| **T1** *Dubliners* | 128 tokens | 64 tokens | Short, self-contained sentences; local coherence is the relevant unit |
| **T2** *Portrait* | 128 tokens | 64 tokens | Free indirect discourse operates at paragraph scale |
| **T3** *Ulysses* (full) | 256 tokens | 128 tokens | Episode-level register shifts are intentional; short windows would pathologically penalise Joyce's own gear-changes |
| **T3-P** *Penelope* | 64 tokens | 32 tokens | Unpunctuated stream-of-consciousness; local rhythm is the relevant unit; no sentence boundaries to anchor longer windows |
| **T4** *Finnegans Wake* | 256 tokens | 128 tokens | Neologism density is high but register is globally uniform; longer windows prevent over-penalisation of local lexical noise |

**Important**: for T3 (full *Ulysses*), the episode-level register variation is a known property of the text. The stylometric centroid $\bar{f}_{T3}$ represents an *average* across very different episodes. The P3 loss will therefore exert softer centroid pull for T3 than for T3-P, which is by design — T3-P has a tight, well-defined stylometric target; T3 does not.

---

## Precomputed Centroids

Both $\bar{e}_T$ and $\bar{f}_T$ are computed once before P3 training begins and held fixed throughout.

**$\bar{e}_T$ computation**: pass all documents in the target tier's training split through the P2-trained model (frozen). Collect mean-pooled final-layer hidden states for each document. Average across all documents. This gives the target tier centroid in the model's own representation space at the end of P2 — the space P3 will operate in.

**$\bar{f}_T$ computation**: compute $f_i$ for all windows across the target tier's training split. Average component-wise. This gives the stylometric centroid of the human-authored source text.

**Why frozen**: if centroids are recomputed dynamically against the model's evolving representations, the target moves with the model and the loss becomes degenerate. The centroid must represent the human-authored source, not the model's current approximation of it.

---

## Hyperparameters

| Parameter | Symbol | Suggested range | Notes |
|---|---|---|---|
| Embedding consistency weight | $\lambda_e$ | 0.1 – 0.5 | Higher values enforce stricter directional consistency; tune against generation fluency |
| Stylometric smoothness weight | $\lambda_s$ | 0.1 – 0.5 | Higher values suppress register lurching; tune against diversity of outputs |
| Initial embedding schedule weight | $\alpha_0$ | 1.0 | Fixed; $\alpha$ decays from this value |
| Initial stylometric schedule weight | $\beta_0$ | 1.0 | Fixed; $\beta$ grows toward this value |
| Curriculum crossover rate | $\gamma$ | 1e-4 – 1e-3 | Calibrate to embedding loss plateau; lower = slower handoff to stylometric loss |
| Embedding loss plateau threshold | $\epsilon_e$ | 1e-3 | Rate of improvement below which crossover acceleration is triggered |

All hyperparameters are logged per run. Sensitivity analysis across $\lambda_e$ and $\lambda_s$ is reported in the evaluation for G-T3-P (Penelope), which serves as the P3 showcase sub-experiment due to its tight, well-defined stylometric target.

---

## Relationship to the Adversarial Evaluation

P3 is a training objective, not an evaluation metric. Its purpose is to produce generators that are stronger adversarial attackers — not to directly measure misattribution rate.

The connection is causal: a generator that passes P3 should exhibit lower function-word KL divergence from the source corpus at matched training steps than a generator trained under P1+P2 alone, because P3 penalises the stylometric drift that KL divergence measures. This makes it a stronger attacker against D1 (classical) classifiers, which are trained on exactly those surface features.

Whether P3-trained generators are also stronger attackers against D2 (neural) classifiers is an open empirical question and one of the secondary findings the evaluation matrix is designed to answer.

P3 also has a direct interpretation under μp → UP: it is the mechanism that makes the arrow a **morphism** rather than merely a function. A function maps inputs to outputs. A morphism preserves structure — in this case, the stylistic structure of the target tier across the full length of the generated sequence. P3 enforces that the transformation does not merely produce locally correct outputs but maintains compositional coherence from the first window to the last.

---

## Implementation Notes

P3 is implemented in `generation/p3_loss.py`. The loss is computed outside the model's autoregressive forward pass — windows are extracted from completed generated sequences, not computed token-by-token during decoding. This means P3 operates as a sequence-level training signal rather than a token-level one, which is consistent with its role as a global coherence constraint.

Gradient flow: $\mathcal{L}_e$ backpropagates through the mean-pooled hidden states into the model's final layers. $\mathcal{L}_s$ does not backpropagate through the model — $f_i$ is computed from surface text using non-differentiable feature extractors. Its role is regularisation via the training loop, not gradient-based optimisation. This is intentional: P3 should not optimise the model to produce text that games the feature extractor. It should optimise the model to produce text that a human reading it would recognise as belonging to the target tier.

---

## Connection to Wake2vec

The direction consistency loss concept originated in Wake2vec, where P3 enforces morpheme compositional coherence on *Finnegans Wake*'s neologisms — units that have no ground truth in standard vocabulary.

MollyLab's P3 is a distinct formulation addressing a different problem. Wake2vec's μ-units are non-standard tokens whose internal composition must be enforced. MollyLab's μ-units are standard English tokens whose *sequential composition* must produce directional coherence in representation and stylometric space. The formula travels; the loss function does not.

| | Wake2vec P3 | MollyLab P3 |
|---|---|---|
| **Unit of concern** | Individual morpheme tokens | Sequential windows of standard text |
| **What is enforced** | Internal token composition coherence | Trajectory coherence across windows |
| **Primary signal** | Morpheme group structure | Embedding + stylometric joint trajectory |
| **Target text** | *Finnegans Wake* | *Ulysses* and full Joycean arc |

Both are proper morphisms in their respective senses. Neither is a lookup table.

---

*From the smallest unit of meaning, the whole voice rises.*

$$\mu p \rightarrow \text{UP}$$
