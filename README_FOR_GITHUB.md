# Precise CDR Position Control in Antibody Sequence Generation

**Anonymous Repository for Peer Review**

This repository contains the complete code and data for the manuscript:

> **"Precise CDR Position Control in Antibody Sequence Generation Using Conditional Deep Generative Models"**
> Submitted to *PLOS ONE* (October 2025)

---

## 📋 Repository Contents

```
.
├── manuscript/           # Manuscript files (Markdown source)
│   ├── main_manuscript.md
│   ├── cover_letter.md
│   └── references.md
├── figures/              # All manuscript figures (300 DPI)
├── supplementary/        # Supporting Information files
│   ├── S1_Table_data_cleaning.csv
│   ├── S2_Table_vocabulary_mapping.csv
│   ├── S3_Table_amino_acid_frequencies.csv
│   └── S4_Table_hyperparameter_sensitivity.csv
├── scripts/              # Code for data processing, training, and evaluation
│   ├── data_preprocessing.py
│   ├── cdr_position_marking.py
│   ├── model_mamba.py
│   ├── model_transformer.py
│   ├── train.py
│   ├── evaluate.py
│   └── generate_sequences.py
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

**Note:** Model checkpoint files (88.6M Mamba, 50.5M Transformer) are hosted separately on Zenodo due to size constraints.

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n antibody-gen python=3.9
conda activate antibody-gen

# Install dependencies
pip install -r requirements.txt

# Install Mamba SSM (requires CUDA)
pip install mamba-ssm
```

### 2. Data Preparation

Download raw data from public sources:

- **OAS (Observed Antibody Space):** https://opig.stats.ox.ac.uk/webapps/oas
- **SAbDab (Structural Antibody Database):** http://opig.stats.ox.ac.uk/webapps/newsabdab

Run preprocessing pipeline:

```bash
python scripts/data_preprocessing.py \
  --oas_dir ./data/raw/oas \
  --sabdab_dir ./data/raw/sabdab \
  --output_dir ./data/processed
```

### 3. CDR Position Marking

Apply our novel position marking algorithm:

```bash
python scripts/cdr_position_marking.py \
  --input ./data/processed/sequences.csv \
  --output ./data/marked/sequences_with_markers.csv
```

### 4. Model Training

**Mamba model:**

```bash
python scripts/train.py \
  --model mamba \
  --config configs/mamba_config.yaml \
  --data ./data/marked/sequences_with_markers.csv \
  --output ./checkpoints/mamba/
```

**Transformer model:**

```bash
python scripts/train.py \
  --model transformer \
  --config configs/transformer_config.yaml \
  --data ./data/marked/sequences_with_markers.csv \
  --output ./checkpoints/transformer/
```

### 5. Sequence Generation

Generate antibody sequences with CDR position control:

```bash
python scripts/generate_sequences.py \
  --model_path [ZENODO_CHECKPOINT_PATH] \
  --model_type mamba \
  --num_sequences 100 \
  --condition hydrophobic,positive \
  --output ./generated/sequences.fasta
```

### 6. Evaluation

Reproduce manuscript results:

```bash
python scripts/evaluate.py \
  --model_path [ZENODO_CHECKPOINT_PATH] \
  --test_data ./data/processed/test_set.csv \
  --output_dir ./results/
```

---

## 📊 Key Results

| Model | Val Loss | CDR Position Accuracy | Inference Speed | Parameters |
|-------|----------|----------------------|-----------------|------------|
| **Mamba** | 0.4636 | **100.0%** | 3.04 seq/s | 88.6M |
| **Transformer** | 0.6187 | 98.0% | **6.49 seq/s** | 50.5M |

**Conditional Control (p<0.001):**
- ✅ Hydrophobicity: Cohen's d = 1.05 (Mamba), 2.52 (Transformer)
- ✅ Charge: Cohen's d = 0.73 (Mamba), 1.06 (Transformer)

---

## 🔬 Model Checkpoints

Pre-trained model weights are available on **Zenodo**:

- **Mamba checkpoint (88.6M params):** [ZENODO_LINK_TO_BE_INSERTED]
- **Transformer checkpoint (50.5M params):** [ZENODO_LINK_TO_BE_INSERTED]

**Files included in Zenodo archive:**
```
antibody_models_v1.0.zip
├── mamba_epoch8_valloss0.4636.pth
├── transformer_epoch13_valloss0.6187.pth
├── vocab.json
├── config_mamba.yaml
└── config_transformer.yaml
```

---

## 📖 Citation

If you use this code or models in your research, please cite:

```bibtex
@article{jiang2025antibody,
  title={Precise CDR Position Control in Antibody Sequence Generation Using Conditional Deep Generative Models},
  author={Jiang, Pan},
  journal={PLOS ONE},
  year={2025},
  note={Under review}
}
```

---

## 📄 License

This code is released under the **MIT License** for academic and non-commercial use.

The trained models are released under **CC BY 4.0** license.

---

## 🙏 Acknowledgments

- **Data sources:** OAS, SAbDab
- **Frameworks:** PyTorch, Mamba SSM
- **Compute:** Tsinghua University

---

## 📧 Contact (After Publication)

For questions regarding this work, please contact the corresponding author via the information provided in the published manuscript.

**Note:** This is an anonymous repository for peer review. Author identities and full contact information will be revealed upon publication acceptance.

---

**Last Updated:** October 2025
**Repository Status:** Anonymous submission for peer review
**Manuscript Status:** Under review at PLOS ONE
