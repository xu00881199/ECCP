# ECCP

ECCP is an evidence-conditioned clinical planning model for radiology report generation. This repository contains the core ECCP model and the training/evaluation entry points for IU X-Ray and MIMIC-CXR experiments.

## Repository Scope

Included runtime components:

- `models/acrp/`: ECCP network, anatomy evidence tokenizer, and interventional evidence estimator.
- `datasets/clinical_plan_builder.py`: clinical plan construction from reports.
- `datasets/tokenizers.py`: report tokenizer and vocabulary loading.
- `train_eccp_iu.py`: IU X-Ray training/evaluation entry point.
- `train_eccp_mimic.py`: MIMIC-CXR training/evaluation entry point.
- `tools/clinical_consistency_metrics.py`: evidence-plan-report consistency metrics.

Not included: datasets, checkpoints, generated outputs, notebooks, ablation sweep logs, visualization-only scripts, paper drafts, and legacy baseline entry points.

## Environment

```bash
conda env create -f env/environment_eccp_minimal.yml
conda activate eccp
pip install -r requirements.txt
```

## Data Layout

Keep datasets outside git. Example paths:

```text
/root/autodl-tmp/ECCP/data/iu_xray/annotation.json
/root/autodl-tmp/ECCP/data/iu_xray/images300
/root/autodl-tmp/ECCP/data/mimic_cxr/mimic_ammrg_annotation.json
/root/autodl-tmp/physionet.org/files/mimic-cxr-jpg/2.0.0/images300
```

## MIMIC-CXR

```bash
python train_eccp_mimic.py \
  --anno_path /path/to/mimic_ammrg_annotation.json \
  --data_dir /path/to/mimic-cxr-jpg/images300 \
  --output_dir output/eccp_mimic \
  --visual_backbone resnet18 \
  --pretrained_backbone \
  --max_images 2 \
  --epochs 30 \
  --batch_size 32 \
  --device cuda:0
```

Evaluation:

```bash
python train_eccp_mimic.py \
  --anno_path /path/to/mimic_ammrg_annotation.json \
  --data_dir /path/to/mimic-cxr-jpg/images300 \
  --checkpoint_path output/eccp_mimic/eccp_mimic_best.pt \
  --eval_only \
  --eval_split test \
  --device cuda:0
```

## IU X-Ray

```bash
python train_eccp_iu.py \
  --anno_path /path/to/iu_xray/annotation.json \
  --data_dir /path/to/iu_xray/images300 \
  --output_dir output/eccp_iu \
  --visual_backbone resnet18 \
  --pretrained_backbone \
  --max_images 2 \
  --epochs 30 \
  --batch_size 32 \
  --device cuda:0
```
