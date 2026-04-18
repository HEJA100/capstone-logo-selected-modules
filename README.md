# Capstone LOGO Selected Modules

This repository contains the curated code used in my BMI5111 Capstone Project at the National University of Singapore (NUS), based on selected modules from the LOGO framework for non-coding genome interpretation.

The project focuses on reproducing and extending representative tasks from the original LOGO study, with emphasis on promoter identification, selected epigenomic / chromatin-feature-related workflows, and non-coding variant prioritization.

This repository is a **cleaned capstone submission version** rather than a full mirror of the original upstream LOGO repository. Large raw datasets, model checkpoints, TFRecords, logs, and intermediate outputs are intentionally excluded.

---

## 1. Project Context

LOGO is a deep learning / genome language model framework designed for interpretation of non-coding genomic sequences. The original work combines transformer-style language modeling ideas with convolutional and task-specific prediction modules to improve sequence representation learning and functional interpretation in the non-coding genome.

In this capstone project, I did not attempt to package the entire upstream LOGO repository as-is. Instead, I focused on a set of selected modules that were most relevant to my project scope and report writing. The main purpose of this repository is therefore to present a **selected-module reproduction and extension effort** in a form that is concise, reviewable, and aligned with the capstone submission.

The code retained here reflects the parts of the LOGO framework that I actually worked with for:
- reproduction,
- selected follow-up experiments,
- code organization,
- report-aligned interpretation,
- and academic submission.

---

## 2. Upstream Reference

This capstone work is based on the LOGO framework introduced in:

**Meng Y, Huang L, Huang H, Tang H, Zhang N, Yang H, Wu J, Mu F.**  
*Integrating convolution and self-attention improves language model of human genome for interpreting non-coding regions at base-resolution.*  
*Nucleic Acids Research* (2022).  
https://doi.org/10.1093/nar/gkac326

Original upstream repository:  
https://github.com/melobio/LOGO

This repository should therefore be understood as a **capstone-oriented, selected-module code release derived from the upstream LOGO project**, not as a replacement for the original official repository.

---

## 3. Purpose of This Repository

The purpose of this repository is to provide a clean and academically reviewable code snapshot for my capstone project.

More specifically, this repository is intended to:
1. document the selected LOGO modules used in my project;
2. preserve the scripts relevant to my reproduction work;
3. retain selected extensions and experiment scripts developed during the capstone;
4. keep the project lightweight enough for GitHub-based academic review;
5. align the code release with the methodological scope discussed in the capstone report.

Because of this submission-oriented goal, the repository is intentionally narrower than the full upstream project and focuses on the modules, scripts, and utilities most relevant to the work I actually carried out.

---

## 4. Scope of the Capstone Work

This capstone project focuses on selected LOGO-related tasks for non-coding genome interpretation, especially those that are representative of sequence understanding and downstream functional interpretation.

The core scope covered by this repository includes:

### 4.1 Promoter Identification
A major part of the project focused on LOGO-based promoter prediction. This includes scripts related to promoter classification across different promoter subsets such as:
- BOTH
- TATA_BOX
- NO_TATA_BOX

This part of the repository also retains capstone-specific experiment scripts associated with:
- sequence-oriented settings,
- knowledge-enabled settings,
- knowledge-related ablation variants,
- locality-related architectural variants,
- selected comparison-oriented workflows for promoter modeling.

### 4.2 Selected Epigenomic / Chromatin-feature-related Workflows
This repository also retains selected scripts from epigenomic and chromatin-feature-related workflows. These are included because they were part of the broader selected-module exploration and help document how LOGO-style sequence modeling connects to downstream functional prediction tasks.

### 4.3 Non-coding Variant Prioritization
Another important part of the project concerns selected scripts for non-coding variant prioritization. These scripts are included as part of the capstone reproduction and code organization effort, especially for selected C2P-style or related workflows derived from the LOGO framework.

---

## 5. Capstone-specific Contributions

Compared with the upstream LOGO repository, this capstone repository emphasizes the parts that were actually used for my own reproduction, analysis, and report writing.

The capstone-specific work reflected here mainly includes:

- organizing selected LOGO modules into a cleaner project structure for academic submission;
- reproducing selected promoter identification and variant prioritization workflows;
- preserving promoter-related experiment scripts for focused ablation-style comparisons;
- retaining selected scripts related to chromatin / epigenomic workflows;
- packaging only the code files needed for review, rather than the full original large-scale project artifacts.

In particular, the promoter-related part of the repository reflects a more focused experimental exploration than a simple one-to-one copy of the upstream code. It includes scripts associated with:
- knowledge-related promoter experiments,
- locality-related promoter experiments,
- selected architecture and comparison-oriented experiment organization.

This means that the repository not only reflects reproduction of selected upstream functionality, but also the capstone-specific process of narrowing, organizing, and extending those selected components for a more coherent project narrative.

---

## 6. Repository Structure

The main repository structure is shown below:

```text
.
├── 02_LOGO_Promoter/
├── 03_LOGO_EPI/
├── 04_LOGO_Chromatin_Feature/
├── 05_LOGO_Variant_Prioritization/
├── bgi/
├── setup_locality_ablation_jobs.sh
└── README.md
