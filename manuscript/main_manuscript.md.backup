# Precise CDR Position Control in Antibody Sequence Generation Using Conditional Deep Generative Models

## Authors
Pan Jiang^1,*^

^1^ Tsinghua University, Beijing, China

\* Corresponding author: jiangp21@tsinghua.org.cn

---

## Abstract

**Background:** Antibody design relies critically on the complementarity-determining regions (CDRs), which determine binding specificity and affinity. Existing deep learning approaches for antibody sequence generation lack precise control over CDR positioning within sequences, limiting their practical utility.

**Methods:** We developed a novel data preprocessing method that inserts CDR markers at their exact positions within antibody sequences during training. We trained two architectures—Mamba (88.6M parameters) and Transformer (50.5M parameters)—on 10.88 million real antibody sequences with conditional generation based on CDR3 physicochemical properties (hydrophobicity, charge, aromaticity, polarity).

**Results:** Our method achieved 100% CDR position accuracy for Mamba and 98% for Transformer. Mamba demonstrated superior training quality (validation loss 0.4636 vs 0.6187, 25.1% lower) and faster convergence (8 vs 13 epochs). Transformer showed 2.13× faster inference speed and stronger conditional control. Hydrophobicity and charge conditioning were statistically significant (p<0.001, Cohen's d >0.7), while aromaticity and polarity control require further optimization.

**Conclusions:** To our knowledge, this work presents the first method to achieve precise CDR positioning in generated antibody sequences through explicit position marking, with systematic comparison of modern architectures. The models and code are publicly available, providing a practical tool for computational antibody design.

**Keywords:** Antibody design, CDR generation, Deep learning, Mamba, Transformer, Conditional generation

---

## Introduction

### Background

Therapeutic antibodies represent a rapidly growing class of biologics, with over 100 FDA-approved antibody drugs and hundreds more in clinical trials [1,2]. The binding specificity and affinity of antibodies are primarily determined by their complementarity-determining regions (CDRs), particularly CDR3, which exhibits the highest sequence diversity and makes the most significant contacts with antigens [3,4]. Traditional antibody discovery relies on experimental screening of large libraries, which is time-consuming and costly [5].

Deep learning has emerged as a promising approach for *in silico* antibody design [6-8]. Recent models such as AbLang [9], IgLM [10], and RITA [11] can generate antibody-like sequences by learning from millions of natural antibody sequences. However, these methods face a fundamental limitation: they cannot precisely control where CDRs appear within generated sequences. Existing approaches either append CDR information at the sequence end [12] or use global conditioning without positional marking [13], causing the model to lose spatial information about CDR placement.

### Contributions

This work makes four key contributions:

1. **Precise CDR position marking method:** We introduce an algorithm that inserts CDR start/end markers at their exact positions during data preprocessing, achieving 100% position accuracy.

2. **Systematic architecture comparison:** We provide the first comprehensive comparison of Mamba (state space model) and Transformer architectures for antibody generation, revealing complementary strengths.

3. **Conditional generation with validation:** We implement and validate multi-dimensional conditional control based on CDR3 physicochemical properties with rigorous statistical testing.

4. **Large-scale experimental validation:** We train models on 10.88 million real antibody sequences over 117 GPU-hours, ensuring robust results.

### Related Work

**Protein language models:** Large language models pre-trained on protein sequences [14,15] have shown remarkable capabilities in understanding protein structure and function. ESM-2 [16] and ProtGPT2 [8] demonstrate that transformers can learn meaningful protein representations.

**Antibody-specific models:** AbLang [9] trained on 1.5 billion parameters using paired heavy-light chain sequences. IgLM [10] focused on CDR3 infilling tasks, demonstrating region-specific generation but operating on isolated CDR3 regions rather than full-length VH sequences with all three CDRs positioned simultaneously. AntiBERTa [18] adapted BERT architecture for antibody sequences. To our knowledge, none of the existing methods achieve precise positioning control for all CDRs within full-length variable region generation.

**Conditional generation:** CDRH3 generation models [19,20] condition on sequence properties but lack full variable region generation. Our work extends conditional control to complete VH sequences while maintaining CDR position accuracy.

**State space models:** Mamba [21] recently showed competitive performance with transformers at lower computational cost for long sequences. This is its first application to antibody design.

---

## Methods

### Data Collection and Preprocessing

#### Data Sources

We obtained antibody heavy chain variable region (VH) sequences from two public databases:

- **Observed Antibody Space (OAS)** [22]: A comprehensive repository of antibody repertoires from high-throughput sequencing
- **Structural Antibody Database (SAbDab)** [23]: A curated database of antibody structures with annotated CDR regions

#### CDR Definition and Numbering

All antibody sequences were numbered using the **IMGT numbering scheme** [31], which is the standard system employed by both OAS and SAbDab databases. The IMGT system provides consistent CDR definitions across all antibody sequences:

- **CDR1**: IMGT positions 27-38
- **CDR2**: IMGT positions 56-65
- **CDR3**: IMGT positions 105-117

CDR boundaries were extracted directly from the database annotations, which were pre-validated by the OAS/SAbDab pipelines using ANARCI (ANtibody Numbering and Receptor ClassIfication) [24] for automatic numbering. Our position marking algorithm (Algorithm 1) uses these pre-annotated CDR sequences and verifies their presence within the full VH sequence via substring matching, ensuring consistency with the IMGT framework.

**Table 1. Dataset Statistics**

| Metric | Value |
|--------|-------|
| Raw sequences | 11,243,567 |
| Complete records (all CDRs) | 10,876,234 |
| Species distribution | Human (85%), Mouse (12%), Other (3%) |
| Sequencing platform | Illumina HiSeq/NovaSeq |
| Data version | OAS 2023-01, SAbDab 2023-03 |

#### Quality Control

We implemented a four-stage filtering pipeline (Fig 1A):

1. **Completeness check:** Require sequence, CDR1, CDR2, CDR3 fields
2. **Length filtering:** Sequence length 80-200, CDR3 length 5-35 amino acids
3. **Character validation:** Only 20 standard amino acids
4. **Substring matching:** CDRs must exist within full sequence

Based on validation of 1,000,000 records, we achieved 98.8% retention rate (Table S1).

#### Sequence Length Analysis

The cleaned dataset shows typical VH characteristics (Fig 1B):

- Mean sequence length: 122.3 ± 5.6 amino acids (normal distribution)
- Mean CDR3 length: 17.0 ± 4.6 amino acids (right-skewed distribution)
- CDR1/CDR2 lengths: 8.0 ± 0.6 and 7.9 ± 0.8 (low variance)

The high variability in CDR3 length (std=4.6) reflects its biological role in antigen recognition diversity.

### CDR Precise Position Marking Algorithm

#### Problem Formulation

**Challenge:** Enable the model to learn CDR positions within the sequence coordinate system.

**Existing limitations:**
- End-appending: `[sequence][<CDR>][CDR_sequence]` loses positional information
- Global labeling: Uses condition tokens without position markers

**Our solution:** Insert special tokens at the actual CDR positions in preprocessing.

#### Algorithm Design

We developed a reverse-order insertion algorithm (Algorithm 1) that places CDR markers at their true positions while avoiding index shifting.

**Algorithm 1: CDR Position Marking**

```
Input: full_seq, cdr1, cdr2, cdr3
Output: marked_seq with position markers

1. Initialize: result ← full_seq, insertions ← []

2. Locate CDRs (record position and length):
   - CDR3: pos3 ← result.rfind(cdr3)  # rightmost search
   - CDR2: pos2 ← result.rfind(cdr2)
   - CDR1: pos1 ← result.find(cdr1)    # leftmost search

3. Store insertions:
   insertions ← [(pos1, len1, tags1), (pos2, len2, tags2), (pos3, len3, tags3)]

4. Sort in reverse order:
   insertions.sort(key=position, reverse=True)

5. Insert markers from right to left:
   for (pos, length, start_tag, end_tag) in insertions:
       result ← result[:pos] + start_tag + result[pos:pos+length] +
                end_tag + result[pos+length:]

6. Return result
```

**Key insight:** Reverse-order insertion (step 4-5) prevents position shifting. Inserting from the rightmost position ensures that previously recorded left positions remain valid.

**Example:**

```
Original:
QLVESGGGLVQ...GFTFSSYA...INSGGGST...AADGGYYCLGLEPYEYDF...

Marked:
QLVESGGGLVQ...<cdr1_start>GFTFSSYA<cdr1_end>...
<cdr2_start>INSGGGST<cdr2_end>...<cdr3_start>AADGGYYCLGLEPYEYDF<cdr3_end>...
```

#### Validation

We validated marking accuracy on 500 sequences by:
1. Extracting marked CDR sequences
2. Removing all markers to obtain clean sequence
3. Verifying CDR positions match original annotations

**Results:** Mamba achieved 100% accuracy (500/500), Transformer achieved 98% (490/500). The 10 Transformer errors showed ±3-5 amino acid position shifts, not complete misplacement.

### CDR3 Property Calculation

To enable conditional generation, we compute four physicochemical properties:

#### Hydrophobicity

Based on Kyte-Doolittle scale [25]:

$$H(seq) = \frac{1}{L} \sum_{i=1}^{L} h(aa_i)$$

Classification:
- Hydrophobic: H > 0.5
- Hydrophilic: H < -0.8
- Neutral: -0.8 ≤ H ≤ 0.5

#### Net Charge

At physiological pH 7.4:

$$C(seq) = \frac{1}{L} \left( \sum_{aa \in \{K,R\}} 1 + \sum_{aa \in \{H\}} 0.5 - \sum_{aa \in \{D,E\}} 1 \right)$$

Classification:
- Positive: C > 0.1
- Negative: C < -0.1
- Neutral: -0.1 ≤ C ≤ 0.1

#### Aromaticity

$$A(seq) = \frac{|\{i : aa_i \in \{F, Y, W\}\}|}{L}$$

Classification: Aromatic if A > 0.2

#### Polarity

$$P(seq) = \frac{|\{i : aa_i \in \{S,T,N,Q,Y,C,K,R,D,E\}\}|}{L}$$

Classification: Polar if P > 0.5

These properties influence antibody-antigen binding modes and developability [25,26].

### Vocabulary and Tokenization

Our vocabulary consists of 40 tokens (Table S2):

- Special tokens (4): `<pad>`, `<bos>`, `<eos>`, `<mask>`
- Amino acids (20): A-Y
- CDR markers (6): `<cdr1_start>`, `<cdr1_end>`, etc.
- Condition tokens (12): hydrophobicity (3), charge (3), aromaticity (2), polarity (2)
- Total: 42 → 40 (actual implementation)

**Encoding format:**

```
<bos> <hydrophobic> <positive> <aromatic> <polar>
[sequence with CDR markers] <eos>
```

Maximum sequence length: 256 tokens (covers 99% of sequences after encoding).

**Dataset split:** 80% training (8,700,987), 20% validation (2,175,247).

### Model Architectures

#### Mamba Architecture

Mamba is a state space model (SSM) with linear-time complexity [21]. The core SSM equations:

$$\frac{dx(t)}{dt} = \mathbf{A}x(t) + \mathbf{B}u(t)$$
$$y(t) = \mathbf{C}x(t)$$

**Selective SSM innovation:** Parameters B and C depend on input:

$$\mathbf{B}_k = s_B(x_k), \quad \mathbf{C}_k = s_C(x_k)$$

This enables input-dependent forgetting/retention.

**Hyperparameters (Table 2):**

| Parameter | Value |
|-----------|-------|
| Hidden dimension ($d_{model}$) | 1024 |
| State dimension ($d_{state}$) | 128 |
| Convolution size ($d_{conv}$) | 16 |
| Expansion factor | 6 |
| Number of layers | 4 |
| Dropout | 0.1 |
| **Total parameters** | **88.6M** |

#### Transformer Architecture

We employ a **decoder-only Transformer** architecture (similar to GPT-style models [36]) for autoregressive sequence generation. The architecture uses multi-head self-attention with causal masking [28] to ensure that each position can only attend to previous positions:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}$$

where $M_{ij} = -\infty$ if $i < j$ (causal mask preventing attention to future tokens), and $M_{ij} = 0$ otherwise.

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Hidden dimension ($d_{model}$) | 1024 |
| Attention heads | 8 |
| FFN dimension ($d_{ff}$) | 4096 |
| Number of layers | 4 |
| Dropout | 0.1 |
| **Total parameters** | **50.5M** |

**Key difference:** Transformer has 43% fewer parameters but $O(L^2)$ complexity vs Mamba's $O(L)$.

### Training Strategy

#### Loss Function

Cross-entropy loss with padding masking:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{L_i} \mathbb{1}[y_t^{(i)} \neq \text{<pad>}] \log P(y_t^{(i)} | x_{<t}^{(i)}; \theta)$$

#### Optimization

AdamW optimizer [29] with:

- Learning rate: 2×10^-6^ (maximum), 5×10^-7^ (minimum)
- β~1~ = 0.9, β~2~ = 0.95
- Weight decay: 0.01
- Gradient clipping: norm 1.0

**Note:** The extremely low learning rate was necessary to prevent gradient explosion. Initial trials with 3×10^-6^ and 8×10^-6^ resulted in NaN losses within 5 epochs.

#### Learning Rate Schedule

Warmup + Cosine annealing:

$$\eta_t = \begin{cases}
\eta_{max} \cdot \frac{t}{T_{warmup}} & \text{if } t \leq T_{warmup} \\
\eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t - T_{warmup}}{T_{max} - T_{warmup}}\pi\right)\right) & \text{otherwise}
\end{cases}$$

with $T_{warmup} = 2000$ steps, $T_{max} = 100$ epochs.

#### Training Configuration

- Batch size: 1024 sequences
- Hardware: 6× NVIDIA A100 40GB GPUs
- Mixed precision: Automatic Mixed Precision (AMP) FP16
- Early stopping: patience 3 epochs
- Framework: PyTorch 2.0.1, CUDA 11.8

**Training time:**
- Mamba: 8 epochs × 6.5h = 52 hours
- Transformer: 13 epochs × 5h = 65 hours
- **Total: 117 GPU-hours**

---

## Results

### Training Performance

#### Mamba Training Dynamics

**Table 3. Mamba Training Record**

| Epoch | Train Loss | Val Loss | Status |
|-------|-----------|----------|--------|
| 1 | 0.7892 | 0.5124 | Initial |
| 2 | 0.5234 | 0.4663 | Improving |
| 3 | 0.4951 | 0.4787 | Minor fluctuation |
| 4 | 0.4856 | 0.4720 | Improving |
| 5 | 0.4802 | 0.4681 | Improving |
| 6 | 0.4783 | 0.4776 | Minor increase |
| 7 | 0.4765 | 0.4675 | Recovering |
| **8** | **0.4748** | **0.4636** | **Best** |
| 9 | nan | 1.5048 | Overfitting + NaN |

**Key observations:**
- Training loss decreased normally from epoch 1-8 (0.7892 → 0.4748)
- Validation loss improved steadily (0.5124 → 0.4636, 9.5% improvement)
- **NaN appeared only in epoch 9** during overfitting, when validation loss jumped to 1.5048
- Early stopping correctly selected epoch 8 as the best checkpoint
- The NaN coincided with sharp overfitting rather than training instability

#### Transformer Training Dynamics

**Table 4. Transformer Training Record**

| Epoch | Train Loss | Val Loss | Status |
|-------|-----------|----------|--------|
| 1-12 | 0.7234→0.6270 | 0.7156→0.6239 | Steady descent |
| **13** | **0.6255** | **0.6187** | **Best** |
| 14 | 0.6243 | 0.6198 | Val loss increase → stop |

**Key observations:**
- Both training and validation losses decreased smoothly
- Required 13 epochs to converge (vs 8 for Mamba)
- Final validation loss 0.6187, which is 25.1% higher than Mamba

#### Architecture Comparison

**Table 5. Training Comparison**

| Metric | Mamba | Transformer | Advantage |
|--------|-------|-------------|-----------|
| Best val loss | 0.4636 | 0.6187 | Mamba -25.1% |
| Convergence (epochs) | 8 | 13 | Mamba 62.5% faster |
| Training time | 52h | 65h | Mamba saves 13h |
| Train loss stability | NaN | Normal | Transformer |

### CDR Position Accuracy

We validated CDR positioning on 500 randomly selected validation sequences (Table 6).

**Table 6. CDR Position Accuracy**

| Model | Total | CDR1 | CDR2 | CDR3 | All Correct | Overall |
|-------|-------|------|------|------|-------------|---------|
| Mamba | 500 | 500 | 500 | 500 | 500 | **100.0%** |
| Transformer | 500 | 495 | 498 | 493 | 490 | **98.0%** |

**Error analysis (Transformer's 10 failures):**
- CDR3 position shifts: ±3-5 amino acids
- CDR marker omission: 2 cases
- No complete mispositioning observed

**Conclusion:** Both models successfully learned CDR positioning, with Mamba achieving perfect accuracy.

### Conditional Control Effectiveness

We tested conditional generation with controlled experiments:

- **Experimental group:** Generate 100 sequences with specific property condition
- **Control group:** Generate 100 sequences without condition
- **Statistical tests:** Independent t-test and Cohen's d effect size
- **Starting sequence:** "QLV" for all generations

#### Hydrophobicity Control

**Table 7. Hydrophobicity Control Results**

| Model | Condition | Mean | SD | Δ from control | t-statistic | p-value | Cohen's d |
|-------|-----------|------|-----|----------------|-------------|---------|-----------|
| **Mamba** | Conditioned | -0.294 | 0.521 | **+0.525** | 7.42 | <0.001 | 1.05 (large) |
| | Unconditioned | -0.819 | 0.443 | - | - | - | - |
| **Transformer** | Conditioned | 0.123 | 0.687 | **+1.539** | 17.83 | <0.001 | 2.52 (very large) |
| | Unconditioned | -1.416 | 0.562 | - | - | - | - |

**Interpretation:**
- Both models show highly significant control (p<0.001)
- Transformer exhibits stronger effect (Δ=1.539 vs 0.525)
- Relative improvement: Transformer 108.8%, Mamba 64.1%

#### Charge Control

**Table 8. Charge Control Results**

| Model | Condition | Mean | SD | Δ from control | t-statistic | p-value | Cohen's d |
|-------|-----------|------|-----|----------------|-------------|---------|-----------|
| **Mamba** | Conditioned | -0.011 | 0.156 | **+0.118** | 5.13 | <0.001 | 0.73 (medium-large) |
| | Unconditioned | -0.129 | 0.168 | - | - | - | - |
| **Transformer** | Conditioned | 0.087 | 0.203 | **+0.206** | 7.49 | <0.001 | 1.06 (large) |
| | Unconditioned | -0.119 | 0.187 | - | - | - | - |

**Interpretation:**
- Transformer shows nearly 2× stronger control (Δ=0.206 vs 0.118)
- Mamba achieves near-neutral charge (-0.011), Transformer reaches positive (+0.087)
- Both statistically significant with large effect sizes

#### Aromaticity and Polarity Control

**Table 9. Aromaticity and Polarity Results**

| Property | Model | Δ from control | p-value | Cohen's d | Significance |
|----------|-------|----------------|---------|-----------|--------------|
| **Aromaticity** | Mamba | +0.023 | 0.077 | 0.25 | ✗ Not significant |
| | Transformer | +0.034 | 0.035 | 0.30 | * Weak (p<0.05) |
| **Polarity** | Mamba | +0.020 | 0.199 | 0.18 | ✗ Not significant |
| | Transformer | +0.027 | 0.097 | 0.24 | ✗ Not significant |

**Analysis:** Aromaticity and polarity control did not reach strong significance (p<0.01 threshold). Possible reasons:
1. Class imbalance (aromatic 18% vs non-aromatic 82%)
2. Higher feature complexity requiring more training
3. Threshold values may need optimization

### Inference Performance

We measured generation speed and memory usage under controlled conditions (Fig 3).

**Table 10. Inference Performance**

| Model | Time (50 seq) | ms/seq | Throughput | Peak Memory | Memory vs Mamba |
|-------|---------------|--------|------------|-------------|-----------------|
| Mamba | 16.42s | 328.5 | 3.04 seq/s | 3569 MB | Baseline |
| Transformer | 7.70s | 154.1 | 6.49 seq/s | 2588 MB | -27.5% |
| **Speedup** | **0.47×** | **0.47×** | **2.13×** | - | - |

**Unexpected finding:** Transformer is 2.13× faster than Mamba, contrary to theoretical expectations (Mamba should be faster with O(n) vs O(n²) complexity).

**Reasons:**
1. Sequence length (256) below Mamba's advantage threshold (~1000 tokens)
2. PyTorch's highly optimized Transformer kernels (Flash Attention)
3. Mamba implementation (`mamba_ssm`) still maturing
4. Modern GPUs optimized for matrix multiplication (Transformer's core operation)

### Sequence Quality

#### Diversity

We assessed mode collapse by counting unique sequences (Table 11).

**Table 11. Sequence Diversity**

| Model | Condition | Generated | Unique | Diversity |
|-------|-----------|-----------|--------|-----------|
| Mamba | Unconditioned | 100 | 96 | 96.0% |
| | Hydrophobic | 100 | 97 | 97.0% |
| | Positive charge | 100 | 96 | 96.0% |
| | **Average** | - | - | **96.3%** |
| Transformer | Unconditioned | 100 | 97 | 97.0% |
| | Hydrophobic | 100 | 98 | 98.0% |
| | Positive charge | 100 | 96 | 96.0% |
| | **Average** | - | - | **97.0%** |

**Conclusion:** Both models maintain high diversity (>95%), indicating no mode collapse.

#### Amino Acid Composition

Generated sequences show realistic amino acid distributions (Table S3). Top frequencies:
- G (12.16%), S (10.70%), V (7.92%), A (7.66%), T (6.89%)

These match known VH region composition [27], with high G/S content supporting CDR flexibility.

---

## Discussion

### Principal Findings

This study presents the first deep learning method to achieve precise CDR positioning in generated antibody sequences, with 100% accuracy for Mamba. Our systematic comparison reveals complementary strengths between Mamba and Transformer architectures: Mamba excels in training quality and CDR positioning, while Transformer offers faster inference and stronger conditional control.

### CDR Position Control Innovation

The key innovation—inserting CDR markers at true positions during preprocessing—overcomes a fundamental limitation of existing methods. Previous approaches [9-11] either discard positional information or rely on the model to implicitly learn CDR locations, resulting in inconsistent positioning. Our explicit marking ensures the model learns CDR spatial relationships within the framework region coordinate system.

The perfect 100% accuracy achieved by Mamba likely stems from its sequential processing nature. State space models maintain a continuous hidden state that naturally tracks position, whereas Transformer's attention mechanism may occasionally "confuse" similar subsequences when determining exact boundaries.

### Architecture Trade-offs

Our results reveal an unexpected discrepancy between theoretical and practical performance:

**Theoretical expectation:** Mamba's O(n) complexity should enable faster inference than Transformer's O(n²).

**Observed reality:** Transformer is 2.13× faster at sequence length 256.

This highlights the importance of considering implementation maturity and hardware optimization, not just algorithmic complexity. For production deployment at length <512, Transformer currently offers better throughput. For research applications prioritizing sequence quality, Mamba's 25% lower validation loss is substantial.

**Recommendation:**
- **High-throughput screening:** Transformer (faster, good enough quality)
- **Lead optimization:** Mamba (higher quality, slower acceptable)
- **Long sequences (>512):** Re-evaluate Mamba advantage

### Conditional Control Success and Limitations

Hydrophobicity and charge control achieved highly significant effects (p<0.001, large Cohen's d), validating our approach for these properties. The stronger Transformer performance (2-3× larger Δ) may result from attention mechanisms explicitly modeling pairwise amino acid interactions, which is crucial for charge distribution and hydrophobic clustering.

The failure of aromaticity and polarity control (p>0.05) reveals current limitations:

1. **Class imbalance:** Only 18% of CDR3s are aromatic, providing limited training signal
2. **Feature complexity:** These properties depend on specific 3-4 residue combinations, harder to learn than global averages
3. **Threshold sensitivity:** Binary classification at single thresholds may be too coarse

**Proposed solutions:**
- Oversample minority classes during training
- Use continuous property values instead of discrete categories
- Extend training epochs with property-focused curriculum learning
- Incorporate structure-aware features (predicted secondary structure)

### Comparison with Existing Methods

**Table 12. Method Comparison**

| Method | CDR Position | Conditional | Architecture | Dataset Size |
|--------|--------------|-------------|--------------|--------------|
| AbLang [9] | ✗ | ✗ | LSTM | 15M sequences |
| IgLM [10] | ✗ | Limited | Transformer | 5.6M |
| RITA [11] | ✗ | ✗ | Transformer | 50M proteins |
| AntiBERTa [18] | ✗ | ✗ | BERT | 2.4M |
| **This work** | ✓ **100%** | ✓ **Validated** | **Mamba + Transformer** | **10.9M** |

Our precise CDR positioning represents a unique capability. While AbLang and IgLM generate high-quality sequences, they cannot guarantee CDR placement, limiting downstream structural modeling.

### Limitations

1. **Lack of experimental validation:** Generated sequences have not been synthesized or tested for binding. Computational structure prediction (e.g., AlphaFold2) would strengthen claims.

2. **Incomplete property control:** Aromaticity and polarity require further optimization.

3. **Single-chain focus:** Only VH sequences; paired VH-VL generation would be more practical.

4. **No specificity control:** Cannot specify target antigen; generated antibodies may not bind intended targets.

### Future Directions

**Short-term (3-6 months):**
1. AlphaFold2 structure prediction of generated sequences
2. Improve aromaticity/polarity control with resampling
3. Extend sequence length experiments to validate Mamba scaling

**Medium-term (6-12 months):**
4. Paired VH-VL generation with interface constraints
5. Incorporate antigen sequence/structure conditioning
6. Wet-lab validation of 20-50 generated sequences

**Long-term (1-2 years):**
7. Multi-species antibody generation (mouse, rabbit, camelid)
8. Developability-aware generation (stability, solubility, immunogenicity)
9. Integration with structure-based design tools

### Practical Impact

This work provides a computational tool for early-stage antibody design:

- **Lead generation:** Quickly generate diverse candidates with desired CDR3 properties
- **Library design:** Create focused libraries around hydrophobicity/charge specifications
- **Education:** Teach antibody sequence-structure relationships through interactive generation

The open-source release (code, models, web interface) lowers barriers for researchers without deep learning expertise.

---

## Conclusions

We present a novel deep learning approach for antibody sequence generation with precise CDR position control, validated on 10.88 million sequences. Key achievements include:

1. To our knowledge, this work achieves 100% CDR position accuracy under our full-length VH generation evaluation with explicit position marking
2. Systematic Mamba-Transformer comparison revealing complementary strengths
3. Validated conditional generation for hydrophobicity and charge (p<0.001)
4. Public release of models and tools for community use

This work advances computational antibody design by solving the CDR positioning problem and providing practical tools for researchers. Future integration with structure prediction and experimental validation will further enhance utility for therapeutic development.

---

## Supporting Information

**S1 Table.** Detailed data cleaning statistics showing filtering steps and retention rates

**S2 Table.** Complete vocabulary mapping (40 tokens) with categories: special tokens, amino acids, CDR markers, and condition tokens

**S3 Table.** Amino acid frequency comparison between real sequences and generated sequences (Mamba and Transformer models)

**S4 Table.** Hyperparameter sensitivity analysis showing tested values and optimal configurations for both architectures

**S1 Fig.** CDR3 property distributions (hydrophobicity, charge, aromaticity, polarity) across the training dataset

---

## Data Availability

**Data & Code Availability.** All data necessary to replicate the findings are available as follows.

**Raw sources:** Training sequences were obtained from the Observed Antibody Space (OAS, https://opig.stats.ox.ac.uk/webapps/oas) accessed January 2023, and the Structural Antibody Database (SAbDab, http://opig.stats.ox.ac.uk/webapps/newsabdab) accessed March 2023. Download scripts with exact query parameters are provided in the repository.

**Processed artifacts:** The complete dataset of IMGT/ANARCI-numbered VH sequences with CDR position markers, train/validation split indices (80/20), and computed CDR3 physicochemical property labels are included in the archived materials.

**Code & models:** All training, inference, and evaluation scripts along with model checkpoints are deposited at **Zenodo** with a concept DOI reserved for this submission. For peer review, anonymous access is provided at the following links:
- Code repository (GitHub): [ANONYMOUS_REPO_LINK - to be provided at submission]
- Model checkpoints (Zenodo): [ANONYMOUS_ZENODO_LINK - to be provided at submission]

Upon publication, these anonymous links will be replaced with permanent public DOIs.

**Reproducibility:** The repository includes: (1) data preprocessing and CDR position marking algorithms, (2) PyTorch model implementations, (3) training configuration files, (4) evaluation and statistical analysis scripts, (5) trained model weights (88.6M Mamba and 50.5M Transformer checkpoints), (6) environment specifications (requirements.txt and conda environment.yml), (7) complete reproduction instructions, and (8) Jupyter notebooks with usage examples.

This complies with PLOS ONE's data policy on public availability of the minimal dataset underlying the results and facilitates computational reproducibility.

---

## Acknowledgments

We thank the maintainers of the Observed Antibody Space (OAS) and Structural Antibody Database (SAbDab) for providing publicly accessible antibody sequence data. Computational resources were provided by Tsinghua University.

### Use of Generative AI

During manuscript preparation, we used **Claude** (Anthropic) for language editing and for drafting brief text suggestions in non-technical sections (e.g., cover letter boilerplate, formatting consistency checks). All technical content—including methods, results, statistical analyses, figures, and references—was authored, verified, and is fully accountable by the authors. No AI tools were used for data analysis, model training, experimental design, figure generation, or reference creation. The authors are responsible for the accuracy and integrity of all content. This disclosure follows PLOS policies on AI tools and technologies.

---

## References

1. Crescioli S, Kaplon H, Chenoweth A, Wang L, Visweswaraiah J, Reichert JM. Antibodies to watch in 2024. MAbs. 2024;16(1):2297450.

2. Lu RM, Hwang YC, Liu IJ, Lee CC, Tsai HZ, Li HJ, Wu HC. Development of therapeutic antibodies for the treatment of diseases. J Biomed Sci. 2020;27(1):1.

3. Xu JL, Davis MM. Diversity in the CDR3 region of V(H) is sufficient for most antibody specificities. Immunity. 2000;13(1):37-45.

4. Kunik V, Peters B, Ofran Y. Paratome: an online tool for systematic identification of antigen-binding regions in antibodies based on sequence or structure. Nucleic Acids Res. 2012;40(Web Server issue):W521-W524.

5. Bradbury ARM, Sidhu S, Dübel S, McCafferty J. Beyond natural antibodies: the power of in vitro display technologies. Nat Biotechnol. 2011;29(3):245-254.

6. Alley EC, Khimulya G, Biswas S, AlQuraishi M, Church GM. Unified rational protein engineering with sequence-based deep representation learning. Nat Methods. 2019;16(12):1315-1322.

7. Rives A, Meier J, Sercu T, Goyal S, Lin Z, Liu J, et al. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. Proc Natl Acad Sci U S A. 2021;118(15):e2016239118.

8. Ferruz N, Schmidt S, Höcker B. ProtGPT2 is a deep unsupervised language model for protein design. Nat Commun. 2022;13(1):4348.

9. Olsen TH, Moal IH, Deane CM. AbLang: an antibody language model for completing antibody sequences. Bioinform Adv. 2022;2(1):vbac046.

10. Shuai RW, Ruffolo JA, Gray JJ. IgLM: Infilling language modeling for antibody sequence design. Cell Syst. 2023;14(11):979-989.

11. Hesslow D, Zanichelli N, Notin P, Poli I, Marks D. RITA: a study on scaling up generative protein sequence models. arXiv preprint arXiv:2205.05789. 2022.

12. Akbar R, Robert PA, Pavlović M, Jeliazkov JR, Snapkov I, Slabodkin A, et al. A compact vocabulary of paratope-epitope interactions enables predictability of antibody-antigen binding. Cell Rep. 2021;34(11):108856.

13. Eguchi RR, Anand N, Choma C, Derry A, Kohnert M, Watson JL, et al. IG-VAE: Generative modeling of protein structure by direct 3D coordinate generation. PLoS Comput Biol. 2022;18(6):e1010271.

14. Elnaggar A, Heinzinger M, Dallago C, Rehawi G, Wang Y, Jones L, et al. ProtTrans: toward understanding the language of life through self-supervised learning. IEEE Trans Pattern Anal Mach Intell. 2022;44(10):7112-7127.

15. Brandes N, Ofer D, Peleg Y, Rappoport N, Linial M. ProteinBERT: a universal deep-learning model of protein sequence and function. Bioinformatics. 2022;38(8):2102-2110.

16. Lin Z, Akin H, Rao R, Hie B, Zhu Z, Lu W, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science. 2023;379(6637):1123-1130.

17. Madani A, Krause B, Greene ER, Subramanian S, Mohr BP, Holton JM, et al. Large language models generate functional protein sequences across diverse families. Nat Biotechnol. 2023;41(8):1099-1106.

18. Leem J, Mitchell LS, Farmery JHR, Barton J, Galson JD. Deciphering the language of antibodies using self-supervised learning. Patterns (N Y). 2022;3(7):100513.

19. Amimeur T, Shaver JM, Ketchem RR, Taylor JA, Clark RH, Smith J, et al. Designing feature-controlled humanoid antibody discovery libraries using generative adversarial networks. bioRxiv. 2020.

20. Friedensohn S, Yermanos A, Hoehn KB, Khan TA, Phad GE, Mayer A, et al. Scalable RNA engineering for T cell receptor and chimeric antigen receptor design using a systems pharmacology approach. Cell Syst. 2021;12(12):1126-1140.

21. Gu A, Dao T. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752. 2023.

22. Kovaltsuk A, Leem J, Kelm S, Snowden J, Deane CM, Krawczyk K. Observed antibody space: a resource for data mining next-generation sequencing of antibody repertoires. J Immunol. 2018;201(8):2502-2509.

23. Dunbar J, Krawczyk K, Leem J, Baker T, Fuchs A, Georges G, et al. SAbDab: the structural antibody database. Nucleic Acids Res. 2014;42(Database issue):D1140-D1146.

24. Dunbar J, Deane CM. ANARCI: antigen receptor numbering and receptor classification. Bioinformatics. 2016;32(2):298-300.

25. Kyte J, Doolittle RF. A simple method for displaying the hydropathic character of a protein. J Mol Biol. 1982;157(1):105-132.

26. Raybould MIJ, Marks C, Krawczyk K, Taddese B, Nowak J, Lewis AP, et al. Five computational developability guidelines for therapeutic antibody profiling. Proc Natl Acad Sci U S A. 2019;116(10):4025-4030.

27. Jain T, Sun T, Durand S, Hall A, Houston NR, Nett JH, et al. Biophysical properties of the clinical-stage antibody landscape. Proc Natl Acad Sci U S A. 2017;114(5):944-949.

28. Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, et al. Attention is all you need. In: Advances in Neural Information Processing Systems 30 (NIPS 2017). 2017. p. 5998-6008.

29. Loshchilov I, Hutter F. Decoupled weight decay regularization. In: International Conference on Learning Representations (ICLR). 2019.

30. Micikevicius P, Narang S, Alben J, Diamos G, Garcia D, Ginsburg B, et al. Mixed precision training. In: International Conference on Learning Representations (ICLR). 2018.

31. Lefranc MP, Giudicelli V, Duroux P, Jabado-Michaloud J, Folch G, Aouinti S, et al. IMGT®, the international ImMunoGeneTics information system® 25 years on. Nucleic Acids Res. 2015;43(Database issue):D413-D422.

32. Olsen TH, Boyles F, Deane CM. Observed antibody space: a diverse database of cleaned, annotated, and translated unpaired and paired antibody sequences. Protein Sci. 2022;31(1):141-146.

33. Jumper J, Evans R, Pritzel A, Green T, Figurnov M, Ronneberger O, et al. Highly accurate protein structure prediction with AlphaFold. Nature. 2021;596(7873):583-589.

34. Fu T, Gao W, Xiao C, Yasonik J, Coley CW, Sun J. Differentiable scaffolding tree for molecular optimization. In: International Conference on Learning Representations (ICLR). 2022.

35. Devlin J, Chang MW, Lee K, Toutanova K. BERT: Pre-training of deep bidirectional transformers for language understanding. In: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). 2019. p. 4171-4186.

36. Radford A, Wu J, Child R, Luan D, Amodei D, Sutskever I. Language models are unsupervised multitask learners. OpenAI Blog. 2019;1(8):9.

37. Kingma DP, Ba J. Adam: A method for stochastic optimization. In: International Conference on Learning Representations (ICLR). 2015.

38. Cohen J. Statistical power analysis for the behavioral sciences. 2nd ed. Hillsdale, NJ: Lawrence Erlbaum Associates; 1988.

39. Chiu ML, Goulet DR, Teplyakov A, Gilliland GL. Antibody structure and function: the basis for engineering therapeutics. Antibodies (Basel). 2019;8(4):55.

40. Carter PJ, Lazar GA. Next generation antibody drugs: pursuit of the 'high-hanging fruit'. Nat Rev Drug Discov. 2018;17(3):197-223.

41. Drago JZ, Modi S, Chandarlapaty S. Unlocking the potential of antibody–drug conjugates for cancer therapy. Nat Rev Clin Oncol. 2021;18(6):327-344.

42. Nijkamp E, Ruffolo JA, Weinstein EN, Naik N, Madani A. ProGen2: exploring the boundaries of protein language models. Cell Syst. 2023;14(11):968-978.

43. Cao Y, Shen Y. TALE: Transformer-based protein function Annotation with joint sequence–Label Embedding. Bioinformatics. 2021;37(18):2825-2833.

44. Verkuil R, Kabeli O, Du Y, Wicky BIM, Milles LF, Dauparas J, et al. Language models generalize beyond natural proteins. bioRxiv. 2022.

---

**Manuscript Information**

- Word count (excluding references): ~6,500 words
- Figures: 3 main (Fig 1-3, each with subpanels) + 1 supplementary (Fig S1)
- Tables: 12 main + 4 supplementary
- References: 44 citations
- Submission date: [To be determined]
- Journal: PLOS ONE

---

**END OF MANUSCRIPT**
