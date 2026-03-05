# MollyLab

**Language Models as Adversaries for Stylometry**

> *Molly Bloom will break your classifier. So will Gabriel Conroy. So will H.C. Earwicker. The question is which one breaks it first, and why.*

MollyLab is a research toolkit for **adversarial stylometric evaluation** built around a single author whose career makes that problem maximally hard: James Joyce. It trains compact language models to imitate Joyce's style at different points in his development, then uses those generators to stress-test authorship attribution pipelines — both against non-Joyce prose and against other periods of Joyce himself.

This projects essentially aims to answer two separate questions: 
- the first is a standard adversarial ML question: can a fine-tuned LLM fool a stylometric classifier?
- The second is a harder question that standard stylometry almost never asks: **is a single author even a stable stylistic target?** Joyce, who moved from the spare naturalism of *Dubliners* to the polyglot neologism of *Finnegans Wake* across thirty years, is the sharpest possible test case for that assumption.

---

Before I forget, interlude: [Jorja](https://soundcloud.com/jerryfolkmusic/jorja-2?in=jerryfolkmusic/sets/tracks-3&si=b1fa30eeebd04ffda5e5099863dcc431&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)

---

Stylometric authorship attribution underpins tools used in forensic linguistics, plagiarism detection, literary studies, and academic integrity enforcement. These methods rest on two assumptions: that authors have consistent, machine-detectable stylistic signatures, and that those signatures are stable enough across a corpus to support reliable classification.

Both assumptions are worth stress-testing.

On the first: modern language models can reproduce surface statistical features — function-word distributions, character n-gram profiles, punctuation patterns — with a fidelity no human imitator could match. A compact LLM fine-tuned on a single author's prose is a qualitatively different adversary than any threat the field has previously evaluated against.

On the second: Joyce's published output spans a stylistic range so extreme that it constitutes a natural experiment in intra-author variability. A classifier trained on *Dubliners* confronted with a passage from *Finnegans Wake* — written by the same author — may reject it. That is not an adversarial attack. That is a failure of stylometry's foundational assumption, exposed by the data.

MollyLab treats both problems together: using Joyce's stylistic evolution as the empirical substrate for an adversarial evaluation of how well stylometric methods survive both machine imitation and the author's own development.

---

## Corpus Architecture

The full training and evaluation corpus spans Joyce's four major prose works, treated as four chronological tiers with distinct stylistic registers:

| Tier | Work | Approx. words | Stylistic register |
|---|---|---|---|
| **T1** | *Dubliners* (1914) | ~70,000 | Restrained naturalism; third-person; minimal interiority |
| **T2** | *A Portrait of the Artist as a Young Man* (1916) | ~90,000 | Free indirect discourse; developing stream-of-consciousness |
| **T3** | *Ulysses* (1922) | ~265,000 | Maximum stylistic range across 18 episodes; multiple registers |
| **T3-P** | *Ulysses*, "Penelope" (Episode 18) | ~24,000 | **Named sub-corpus**: pure unpunctuated stream-of-consciousness; extreme case |
| **T4** | *Finnegans Wake* (1939) | ~630,000 | Portmanteau neologism; polyglot collapse; syntax dissolved |

**T3-P (Penelope) is retained as a named sub-corpus** within T3 rather than folded into it. It serves as a distinct sub-experiment — the most extreme single stylistic register in the Joyce canon, and the original motivation for the project. Results for generators trained on T3-P are reported separately from those trained on the full T3 corpus.

The non-Joyce negative class comprises prose fiction contemporaneous with Joyce's output, drawn from public-domain sources, matched approximately for genre (literary fiction) and period. Specific authors TBD; the selection will be documented in `data/control/README.md`.

All text is sourced from public domain editions. Corpus preparation scripts in `data/` handle cleaning, tokenisation, and document-level train/validation/held-out splitting for each tier independently.

**Corpus size imbalance**: T1 (~70k words) is substantially smaller than T4 (~630k words). Classifiers trained on mixed-tier data use stratified sampling to prevent the larger tiers from dominating. Generator training on T1 uses aggressive regularisation and early stopping to reduce memorisation risk; the leakage detection protocol (≥8 consecutive token overlap flagging) is applied uniformly across all tiers.

To note: might also throw Exiles and Occasional, Critical, and Political Writing into the mix just to make it more spicyyy.

---
## Generator Suite

MollyLab trains two classes of generator, giving six generator configurations in total:

### Era-Specific Generators (Primary)

One generator fine-tuned per corpus tier, using **Molly2vec** (embedding-focused adaptation) as the primary regime. These are the main experimental units for the cross-era attack matrix.

| Generator | Training data | PEFT regime |
|---|---|---|
| **G-T1** | *Dubliners* | Molly2vec |
| **G-T2** | *Portrait* | Molly2vec |
| **G-T3** | *Ulysses* (full) | Molly2vec |
| **G-T3-P** | *Ulysses*, "Penelope" only | Molly2vec **and** LoRA (sub-experiment showcase; both regimes) |
| **G-T4** | *Finnegans Wake* | Molly2vec |

### Full-Corpus Generator (Comparison)

One generator fine-tuned on all four tiers concatenated, using **LoRA** as the primary regime. This model absorbs the full stylistic arc and tests whether a single model trained on the complete evolution can attack period-specific classifiers more or less effectively than era-specific generators.

| Generator | Training data | PEFT regime |
|---|---|---|
| **G-full** | T1 + T2 + T3 + T4 | LoRA |

### PEFT Regimes

**Molly2vec (embedding-focused adaptation)**
The transformer backbone is frozen; only the tied embedding matrix is fine-tuned on the target tier. This isolates lexical-level adaptation, shifting vocabulary geometry toward the tier's token distribution without altering syntactic machinery. The hypothesis is that surface stylometric features (function-word profiles, character n-grams) are disproportionately driven by embedding-level changes — making Molly2vec the most efficient attacker against classical D1 classifiers.

**LoRA adaptation**
Low-rank adapters are attached to attention and MLP projection layers; embeddings are held fixed. Rank and alpha are swept over a small grid (r ∈ {4, 8, 16}). Used as the primary regime for G-full and as a comparison regime for G-T3-P, testing whether structural adaptation adds attack power beyond embedding-only changes.

**Independent variable for checkpoint sweeps**: checkpoints are indexed by *function-word KL divergence* from the target tier's source corpus, computed on held-out generated samples. This gives a continuous, semantically meaningful axis of generator strength that is comparable across tiers and regimes. Training steps are logged as a secondary axis only.

**Generation-level leakage control**: each tier is split 80/10/10 into train, validation, and held-out evaluation splits before any generator training. Evaluation uses only prompts drawn from the held-out split. Generated samples reproducing ≥8 consecutive tokens from the training split are flagged and excluded.

Style descriptors logged at each checkpoint: sentence length distribution (mean, std, 95th percentile), punctuation density (commas, em-dashes, ellipses per 100 tokens), lexical diversity (MTLD, type–token ratio over 500-token windows), function-word frequency vector (top-200 English function words), character n-gram entropy (n=3,4).

## Detection Side

Three levels of detector complexity, all trained with strict **document-level splits**:

**D1 — Classical feature-based classifiers**
Function-word frequency profiles (top-200 English function words), character 3- and 4-gram relative frequencies, Burrows's Delta distance to corpus centroid. Classifiers: logistic regression (L2) and linear SVM. These are the primary attack targets and the most widely deployed methods in forensic and literary stylometry.

**D2 — Neural attribution classifier**
A lightweight classifier (two-layer BiLSTM or distilbert-base) operating over stylometric feature sequences rather than raw text. Included to test whether neural classifiers are more or less robust than linear models under identical attack conditions.

**D3 — LLM-as-detector baseline**
A prompted open-weight model (same family as the generators, different checkpoint) asked zero-shot and few-shot whether a passage belongs to a given Joyce period. Tests the "use an LLM to detect LLM text" hypothesis, and whether generator and detector from the same model family systematically blind-spot each other.

Each classifier is trained in **two configurations** corresponding to the two negative-class conditions:

- **Config-A**: positive class = target tier; negative class = non-Joyce prose
- **Config-B**: positive class = target tier; negative class = *other Joyce tiers*

Config-B is the intra-author robustness experiment. A Config-B classifier trained on T1 must distinguish *Dubliners*-era Joyce from *Ulysses*- and *Wake*-era Joyce. Establishing baseline Config-B accuracy (before any adversarial generator is involved) is itself a research contribution. If a classifier already struggles to tell *Portrait* from *Ulysses* without any attack, the subsequent adversarial results mean something different than if it starts from a high baseline.

**Calibration**: all classifiers report predicted probability alongside class labels. ECE (Expected Calibration Error) is reported. Overconfident misattribution is tracked separately from uncertain misattribution as these represent qualitatively different forensic failure modes.

---

## Research Design

### Threat Model

**Attacker goal**: cause a stylometric classifier to assign a generated text the label of the target Joyce period with high confidence, when the ground-truth label is *machine-generated*.

**Ground-truth convention**: generated samples carry the label `machine-generated` regardless of stylistic fidelity. Misattribution means the classifier assigns a period label to a generated sample — not merely that the sample is stylistically plausible.

**Attacker knowledge**:

| Scenario | Attacker knows | Role |
|---|---|---|
| **Black-box** | Only that a stylometric classifier exists | Primary reported metric |
| **Grey-box** | The feature family (e.g. function-word vectors), not the weights | Upper-bound analysis |

**Attacker capability**: PEFT fine-tuning of a ~1B-parameter open-weight LLM on a target tier corpus. No access to classifier training data; no gradient-based attack on the classifier. This is a *transfer attack*: the generator is trained independently and its outputs are passed to the classifier.

### Attack Baselines

**B0 — Unprompted base model**: unmodified LLaMA with no Joyce conditioning. Sets the floor — expected near-zero misattribution.

**B1 — Prompted base model**: base model with 3–5 sentences of the target tier as in-context examples; no weight updates. The "free attacker." If B1 already achieves substantial misattribution against a period classifier, the marginal contribution of fine-tuning must be argued carefully.

---

### Cross-Era Attack Matrix — Condition A (Joyce vs. Non-Joyce)

The primary adversarial evaluation. Each generator attacks each classifier trained under Config-A. Rows = generators; columns = detectors. Each cell reports misattribution rate as a curve over the KL-divergence axis, not a single number.

|  | D1-A Classical | D2-A Neural | D3-A LLM-detector |
|---|---|---|---|
| **B0** (unprompted) | | | |
| **B1** (prompted) | | | |
| **G-T1** (*Dubliners*) | | | |
| **G-T2** (*Portrait*) | | | |
| **G-T3** (*Ulysses*) | | | |
| **G-T3-P** (*Penelope*) | | | |
| **G-T4** (*Wake*) | | | |
| **G-full** (all tiers) | | | |

**Primary hypotheses**:
- **H1**: G-T4 and G-T3-P will achieve the highest misattribution rates against D1-A classifiers, because their extreme stylistic features (neologism, punctuation collapse, stream-of-consciousness lexis) are precisely the features classical stylometry is calibrated on.
- **H2**: G-T1 will be the weakest attacker, because Dubliners-era prose is stylistically closest to literary non-Joyce prose, giving the classifier less Joyce-specific signal to overfit to.
- **H3**: G-full will not dominate era-specific generators; averaging over the full stylistic arc is expected to wash out the extreme features that make individual period generators effective attackers.

---

### Cross-Era Intra-Author Attack Matrix — Condition B (Joyce vs. Joyce)

The intra-author robustness experiment. Classifiers trained under Config-B must distinguish one period of Joyce from other periods of Joyce. Each generator attacks each period-specific classifier.

|  | D1-B: T1 classifier | D1-B: T2 classifier | D1-B: T3 classifier | D1-B: T4 classifier |
|---|---|---|---|---|
| **B0** | | | | |
| **B1** (T1 prompt) | | | | |
| **G-T1** | | | | |
| **G-T2** | | | | |
| **G-T3** | | | | |
| **G-T3-P** | | | | |
| **G-T4** | | | | |
| **G-full** | | | | |

**The diagonal** (G-Tx attacking D1-B: Tx classifier) is the within-era attack — the standard adversarial case, where the generator and classifier are matched to the same period.

**The off-diagonals** are cross-era transfer attacks: does a generator trained on *Finnegans Wake* fool a classifier trained to recognise *Dubliners* prose? If so, the classifier cannot distinguish late-Joyce imitation from genuine early Joyce — a striking result. If not, it reveals that era-specific classifiers retain period-discriminating features that cross-era generators cannot replicate without explicit training on the target period.

**Baseline for Condition B**: before any generator is involved, what is the baseline accuracy of D1-B classifiers on held-out *human* text from a different tier? This is a precondition for interpreting adversarial results. A Config-B classifier that already struggles to tell *Portrait* from *Ulysses* at baseline — without any adversarial pressure — is a different kind of finding than one that starts from a high baseline and is broken only by generator attack.

**Primary hypotheses**:
- **H4**: Baseline Config-B accuracy will be substantially lower for T3 vs. T4 discrimination than for T1 vs. T4 discrimination, reflecting the genuine stylistic distance between adjacent and non-adjacent periods.
- **H5**: G-T4 attacking D1-B: T1 classifier will be the most damaging off-diagonal cell, because the Wake's extreme features are maximally distant from the Dubliners feature space.
- **H6**: G-full attacking any period-specific Config-B classifier will show moderate, non-dominant misattribution — the full-corpus model dilutes the period-specific extremity that makes era-specific generators effective cross-era attackers.

---

### Defence Experiments

Conducted after the attack matrices are complete. Each defence addresses a specific diagnosed failure mode rather than being run speculatively.

**Defence 1 — Adversarial retraining**
*Hypothesis*: classifiers fail because their training distribution contains no machine-generated samples. Adding generated negatives should recover accuracy.
*Protocol*: augment D1 and D2 Config-A training sets with generated samples from each era generator, labelled `machine-generated`. Re-evaluate on fresh held-out sets containing both human text and generated samples not seen during retraining. Critical test: does a classifier retrained on G-T4 samples also recover robustness against G-T1 samples it was *not* retrained on? A defence that only works against the specific generator it trained on is not a general countermeasure.
*Success criterion*: misattribution rate below 20% on the attacking generator; clean accuracy on held-out human text within 5pp of pre-attack baseline; verified not to function as a generic machine-text detector by testing on out-of-distribution generated samples.

**Defence 2 — Perplexity-based detection signal**
*Hypothesis*: fine-tuned generators produce text that is unusually smooth relative to a general-English reference LM — they have overfit to their target distribution in ways that depress perplexity below human-authored levels.
*Protocol*: compute per-sample perplexity under the unmodified base model (as a general-English reference). Add as a feature to D1 classifiers. Key test: does the perplexity signal help specifically against fine-tuned generators (G-Tx) but *not* against B1 (prompted base model with no weight updates)? If so, it is detecting fine-tuning, not stylistic similarity — a more useful and honest claim.

**Defence 3 — Longer-range structural features**
*Hypothesis*: classical stylometry relies on local surface features that generators replicate easily. Paragraph-level and discourse-level structure is harder to imitate at generation time.
*Protocol*: add paragraph-length distribution, sentence-initial function-word patterns, and coreference chain density (via spaCy) to D1 feature vectors. Evaluate marginal accuracy gain under each generator attack. Report whether the benefit is uniform across all generators or concentrated in specific era-generator/classifier combinations.

**What counts as success**: a defence succeeds if it reduces misattribution to below 20% on the generator that broke the original classifier, without reducing clean accuracy on human-authored text by more than 5pp. A defence that works by rejecting all LLM output regardless of style or period is recorded as a *machine-text detector* result, not a *robust stylometric* result — these are different and incompatible claims.

---
### Representation Analysis

An analytic layer explaining *why* attacks succeed or fail across tiers, complementing the empirical matrices.

- **Embedding drift by tier**: cosine distance and norm change from base-model initialisation, tracked per checkpoint for each era generator. Do Wake-trained embeddings drift further than Dubliners-trained ones? Which token classes (function words, punctuation, neologisms) move most, and does the answer differ by tier?
- **Cross-tier embedding comparison**: at matched KL-divergence points, how similar are the embedding spaces of G-T1 and G-T4? Does G-full interpolate between them or collapse toward the largest tier (T4)?
- **Cluster analysis**: k-means clustering at base, mid-training, and final checkpoints. Track whether function-word clusters tighten (suggesting the model is learning stylometrically relevant structure) or whether content-word and neologism clusters fragment (especially relevant for T4, where Joyce's neologisms are not semantically stable units).
- **Linear probing**: probes trained to predict tier membership from intermediate layer activations of generated samples. If a G-T4 sample's activations are identifiable as T4-origin by a probe trained on human text, the generator is encoding period-specific structure internally. If not, it is producing surface features without internal coherence — a meaningful distinction for interpreting why the attack succeeds.

---

## Project Structure

```
mollylab/
│
├── data/
│   ├── t1_dubliners/            # Corpus prep and splits
│   ├── t2_portrait/
│   ├── t3_ulysses/
│   │   └── penelope/            # T3-P sub-corpus, separately tracked
│   ├── t4_wake/
│   └── control/                 # Non-Joyce negative class; selection documented
│
├── generation/
│   ├── molly2vec/               # Embedding-focused PEFT regime
│   ├── lora/                    # LoRA PEFT regime
│   ├── train.py                 # Unified training entry point (--tier, --regime flags)
│   └── sample.py                # Unified sampling with leakage detection
│
├── stylometry/
│   ├── features/                # Function-word, n-gram, discourse feature extractors
│   ├── classifiers/             # D1 (classical), D2 (neural), D3 (LLM-detector)
│   └── evaluate.py              # Calibrated evaluation with ECE reporting
│
├── attacks/
│   ├── run_attack.py            # Single attack run (generator × classifier × condition)
│   └── sweep.py                 # Full matrix sweep; outputs result tables
│
├── defences/
│   ├── adversarial_training/    # Defence 1: augmented retraining
│   ├── perplexity_feature/      # Defence 2: reference-LM perplexity as feature
│   └── structural_features/     # Defence 3: paragraph/discourse feature extension
│
├── analysis/
│   ├── embedding_drift.py       # Per-checkpoint drift tracking, cross-tier comparison
│   ├── cluster_analysis.py      # k-means over embedding space by checkpoint
│   ├── probing.py               # Linear probes for tier membership
│   └── style_descriptors.py     # KL divergence, MTLD, punctuation density logging
│
├── notebooks/                   # Worked examples; result visualisation
│
└── docs/
    ├── background.md            # Literature review
    ├── corpus_selection.md      # Negative class selection rationale
    └── threat_model.md          # Extended threat model discussion
```
---

## Quickstart

*Full documentation in `docs/`. Minimal walkthrough below.*

```bash
git clone https://github.com/your-username/mollylab
cd mollylab
pip install -e ".[dev]"
```

**Prepare all corpus tiers:**
```bash
python data/prepare_all.py \
    --dubliners path/to/dubliners.txt \
    --portrait path/to/portrait.txt \
    --ulysses path/to/ulysses.txt \
    --wake path/to/wake.txt \
    --output data/
```

**Train an era-specific generator:**
```bash
python generation/train.py \
    --tier t4_wake \
    --regime molly2vec \
    --base_model TinyLlama/TinyLlama-1.1B \
    --output_dir checkpoints/g-t4-molly2vec/
```

**Train the full-corpus generator:**
```bash
python generation/train.py \
    --tier all \
    --regime lora \
    --lora_r 8 \
    --base_model TinyLlama/TinyLlama-1.1B \
    --output_dir checkpoints/g-full-lora/
```

**Train a classifier (Config-A: Joyce vs. non-Joyce):**
```bash
python stylometry/evaluate.py \
    --positive data/t1_dubliners/ \
    --negative data/control/ \
    --config A \
    --classifier svm \
    --output results/d1-a-t1/
```

**Train a classifier (Config-B: intra-author period discrimination):**
```bash
python stylometry/evaluate.py \
    --positive data/t1_dubliners/ \
    --negative data/t3_ulysses/ data/t4_wake/ \
    --config B \
    --classifier svm \
    --output results/d1-b-t1/
```

**Run a single attack:**
```bash
python attacks/run_attack.py \
    --generator checkpoints/g-t4-molly2vec/ \
    --classifier results/d1-a-t1/ \
    --output results/attacks/g-t4-vs-d1-a-t1/
```

**Run the full cross-era matrix sweep:**
```bash
python attacks/sweep.py \
    --generators checkpoints/ \
    --classifiers results/ \
    --output results/matrix/
```

---

## Evaluation Outputs

Each attack run produces:

- **Misattribution rate curve** over the KL-divergence axis (primary metric)
- **Classifier confidence distribution** on misattributed samples, with ECE
- **Style descriptor distance** from target tier source corpus
- **Confusion matrices** per checkpoint
- **Cross-era transfer summary**: misattribution rate of each generator against classifiers it was *not* era-matched to

Matrix sweep outputs produce summary tables for both Condition A and Condition B, formatted for direct inclusion in dissertation figures.

---

## Background and Related Work

MollyLab sits at the intersection of four active areas:

- **Authorship attribution and stylometry**: Burrows's Delta (2002), function-word profiling, character n-gram methods, and more recent neural attribution approaches. The intra-author stability assumption is rarely examined explicitly; MollyLab provides empirical pressure on it.
- **Adversarial ML and robustness evaluation**: the attack–defence framing is adapted from adversarial example literature. Stylometric classifiers have not been systematically evaluated under transfer attacks from fine-tuned generators.
- **Parameter-efficient fine-tuning**: LoRA (Hu et al., 2022) and embedding-layer adaptation as mechanisms for style-specific LLM adaptation at low compute cost.
- **LLM-generated text detection**: the complementary problem. MollyLab contributes evidence about the generator strength threshold at which standard attribution methods fail, and whether detection signals (perplexity, structural features) remain viable countermeasures.
- **Computational stylistics**: the Joycean corpus has a documented quantitative stylistic arc. MollyLab is the first project to use that arc as an experimental structure for adversarial ML evaluation.

Fuller discussion and references in `docs/background.md`.

---

## Licence

MIT. See `LICENSE`.

---

## Citation

```bibtex
@misc{biehle2025mollylab,
  author       = {Kyri},
  title        = {{MollyLab}: Language Models as Adversaries for Stylometry},
  year         = {2025},
  howpublished = {\url{https://github.com/mahb97/MollyLab-Language-Models-as-Adversaries-for-Stylometry}},
```





