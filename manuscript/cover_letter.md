# Cover Letter for PLOS ONE Submission

**Date:** October 25, 2025

**To:** PLOS ONE Editorial Office

**Re:** Submission of Research Article "Precise CDR Position Control in Antibody Sequence Generation Using Conditional Deep Generative Models"

---

Dear Editor,

We are pleased to submit our manuscript entitled "**Precise CDR Position Control in Antibody Sequence Generation Using Conditional Deep Generative Models**" for consideration as a Research Article in PLOS ONE.

## Significance and Novelty

Therapeutic antibody design remains a critical challenge in modern medicine, with traditional experimental approaches being time-consuming and costly. Our work addresses a fundamental limitation in computational antibody generation: existing deep learning methods cannot precisely control where complementarity-determining regions (CDRs) appear within generated sequences.

**Key innovations:**

1. **To our knowledge**, this work achieves **100% CDR position accuracy under our full-length VH generation evaluation with explicit position marking**, whereas prior methods such as **IgLM** [Shuai et al., Cell Systems 2023] focus on **CDR3 infilling** rather than positioning **all three CDRs** in full-length sequences

2. **Systematic comparison of modern architectures** (Mamba state space model vs Transformer), revealing that Mamba achieves 25% lower validation loss while Transformer offers 2.13× faster inference

3. **Validated conditional generation** with statistically significant control (p<0.001) over CDR3 hydrophobicity and charge properties

4. **Large-scale validation** using 10.88 million real antibody sequences and 117 GPU-hours of training

## Suitability for PLOS ONE

This work aligns perfectly with PLOS ONE's scope and standards:

- **Rigorous methodology:** Comprehensive statistical validation (t-tests, Cohen's d effect sizes, 98.8% data retention rate)
- **Reproducibility:** All code, models, and data publicly available; detailed hyperparameters and training configurations provided
- **Broad impact:** Addresses challenges in computational biology, machine learning, and therapeutic development
- **Ethical compliance:** Uses only publicly available datasets (OAS, SAbDab); no human/animal subjects

## Why PLOS ONE

We chose PLOS ONE for several reasons:

1. **Open access philosophy:** Ensures our tools reach the widest possible audience in the research community
2. **Interdisciplinary scope:** Bridges deep learning, structural biology, and drug discovery
3. **Data transparency standards:** Aligns with our commitment to full reproducibility
4. **Rapid peer review:** Important for timely dissemination of computational tools

## Competing Interests

The authors declare no competing financial or non-financial interests.

## Previous Presentation

This work has not been previously published or submitted elsewhere. Preliminary results were not presented at conferences.

## Author Contributions (CRediT)

- **Pan Jiang:** Conceptualization, Methodology, Software, Validation, Formal Analysis, Investigation, Data Curation, Writing – Original Draft, Writing – Review & Editing, Visualization

## Suggested Reviewers

We respectfully suggest the following potential reviewers based on their expertise in antibody modeling and deep learning for proteins:

1. **Dr. Charlotte M. Deane**
   - Affiliation: Department of Statistics, University of Oxford, UK
   - Email: deane@stats.ox.ac.uk
   - Expertise: Antibody computational design, protein structure prediction, AbLang development
   - Recent relevant work: Olsen et al. (2022) "AbLang: an antibody language model" in Bioinformatics Advances

2. **Dr. Jeffrey J. Gray**
   - Affiliation: Department of Chemical and Biomolecular Engineering, Johns Hopkins University, USA
   - Email: jgray@jhu.edu
   - Expertise: Protein design, antibody engineering, IgLM development
   - Recent relevant work: Shuai et al. (2023) "IgLM: Infilling language modeling for antibody sequence design" in Cell Systems

3. **Dr. Jianyi Yang**
   - Affiliation: School of Mathematical Sciences, Nankai University, China
   - Email: yangjy@nankai.edu.cn
   - Expertise: Protein structure prediction, deep learning for biomolecules
   - Recent relevant work: Multiple publications on AlphaFold applications and protein language models

4. **Dr. Lucy Colwell**
   - Affiliation: Department of Chemistry, University of Cambridge, UK
   - Email: lucy.colwell@cantab.net
   - Expertise: Machine learning for protein sequences, generative models
   - Recent relevant work: Deep generative models for protein design

5. **Dr. Mohammed AlQuraishi**
   - Affiliation: Department of Systems Biology, Columbia University, USA
   - Email: ma3203@columbia.edu
   - Expertise: Deep learning for protein structure and sequence, protein language models
   - Recent relevant work: Unified rational protein engineering with sequence-based deep learning

## Exclusions

None.

## Data and Code Availability

In accordance with PLOS ONE's data availability policy, we confirm:

- **Code repository:** Anonymous GitHub access is provided for peer review at [ANONYMOUS_REPO_LINK - to be provided at submission]. The repository contains all preprocessing, training, and evaluation code with complete documentation.
- **Model checkpoints:** Anonymous Zenodo access is provided at [ANONYMOUS_ZENODO_LINK - to be provided at submission] with trained weights for both Mamba (88.6M) and Transformer (50.5M) models.
- **Training datasets:** Publicly available from OAS (https://opig.stats.ox.ac.uk/webapps/oas, accessed January 2023) and SAbDab (http://opig.stats.ox.ac.uk/webapps/newsabdab, accessed March 2023). Download scripts with exact query parameters are included in the repository.
- **Supplementary data:** All statistics, intermediate results, and supporting tables provided as CSV and JSON files in the submission package.

Upon publication, all anonymous links will be replaced with permanent public DOIs. A Zenodo concept DOI has been reserved for this submission. This complies with PLOS ONE's data policy requiring availability of the minimal dataset necessary to replicate the findings.

## Funding Statement

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

## Ethics Statement

This study uses only publicly available, de-identified antibody sequence data from high-throughput sequencing repositories. No ethics approval was required.

## Use of Generative AI

We used generative AI (Claude, Anthropic) only for language editing and formatting consistency checks in non-technical sections. All scientific content—including methods, results, data analysis, figures, and references—was produced and verified by the authors. We understand and comply with PLOS policies on AI and authorship.

## Correspondence

For all correspondence regarding this submission, please contact:

**Pan Jiang**
Student
Tsinghua University
Beijing, China
Email: jiangp21@tsinghua.org.cn

We appreciate your consideration of our manuscript and look forward to your response. We are happy to address any questions or provide additional information during the review process.

Sincerely,

**Pan Jiang**
Tsinghua University

---

**Enclosures:**
- Manuscript file (Word/LaTeX)
- Figures (separate files, 300 dpi)
- Supplementary materials
- PLOS ONE submission checklist (completed)

---

**Manuscript Statistics:**
- Word count (main text): ~6,500 words
- References: 44 citations
- Figures: 3 main (Fig 1-3, each with subpanels) + 1 supplementary (Fig S1)
- Tables: 12 main + 4 supplementary
- Supplementary files: Data statistics, vocabulary mapping, amino acid frequencies, hyperparameter analysis
