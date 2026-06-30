import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from datasets.clinical_plan_builder import (
    ANATOMY_NAMES,
    FINDING_NAMES,
    build_clinical_sample,
    compute_entity_f1,
    compute_evidence_support_rate,
    compute_plan_report_consistency,
    merge_promptmrg_labels,
    _findings_from_text,
)
from datasets.tokenizers import Tokenizer
from models.acrp import ECCPNet

try:
    from utils.engine import compute_scores
except Exception:
    compute_scores = None

try:
    from tools.clinical_consistency_metrics import summarize as summarize_consistency
except Exception:
    summarize_consistency = None


def plan_prompt_to_structured_plan(plan_prompt):
    plan = []
    for finding_idx, state in enumerate(plan_prompt):
        if int(state) <= 0:
            continue
        plan.append(
            {
                "anatomy": "global",
                "finding": FINDING_NAMES[finding_idx],
                "attribute": "none",
                "polarity": "positive",
                "evidence_score": 1.0,
            }
        )
    return plan


def compute_internal_consistency(rows):
    if not rows:
        return {
            "Plan_Coverage": 0.0,
            "Report_Plan_Consistency": 0.0,
            "Unsupported_Rate": 0.0,
            "Missing_Rate": 0.0,
            "Consistency_Micro": {},
        }
    per = []
    totals = {"plan": 0, "report": 0, "overlap": 0, "unsupported": 0, "missing": 0}
    for row in rows:
        plan_findings = set()
        for item in row.get("plan", []):
            if str(item.get("polarity", "positive")) != "negative":
                plan_findings.add(str(item.get("finding")))
        report_findings = set(_findings_from_text(row.get("prediction", "")))
        overlap = plan_findings & report_findings
        unsupported = report_findings - plan_findings
        missing = plan_findings - report_findings
        plan_count = len(plan_findings)
        report_count = len(report_findings)
        per.append(
            {
                "Plan_Coverage": len(overlap) / plan_count if plan_count else 1.0,
                "Report_Plan_Consistency": len(overlap) / report_count if report_count else 1.0,
                "Unsupported_Rate": len(unsupported) / report_count if report_count else 0.0,
                "Missing_Rate": len(missing) / plan_count if plan_count else 0.0,
            }
        )
        totals["plan"] += plan_count
        totals["report"] += report_count
        totals["overlap"] += len(overlap)
        totals["unsupported"] += len(unsupported)
        totals["missing"] += len(missing)
    return {
        "Plan_Coverage": float(np.mean([x["Plan_Coverage"] for x in per])),
        "Report_Plan_Consistency": float(np.mean([x["Report_Plan_Consistency"] for x in per])),
        "Unsupported_Rate": float(np.mean([x["Unsupported_Rate"] for x in per])),
        "Missing_Rate": float(np.mean([x["Missing_Rate"] for x in per])),
        "Consistency_Micro": {
            "plan_coverage": totals["overlap"] / totals["plan"] if totals["plan"] else 1.0,
            "report_plan_consistency": totals["overlap"] / totals["report"] if totals["report"] else 1.0,
            "unsupported_rate": totals["unsupported"] / totals["report"] if totals["report"] else 0.0,
            "missing_rate": totals["missing"] / totals["plan"] if totals["plan"] else 0.0,
        },
    }


def set_seed(seed):
    if seed is None or seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_trainable_scope(model, trainable_scope):
    if trainable_scope == "all":
        return
    for param in model.parameters():
        param.requires_grad = False
    if trainable_scope == "decoder":
        trainable_modules = [
            model.token_embedding,
            model.position_embedding,
            model.decoder,
            model.output_head,
        ]
    else:
        raise ValueError("Unsupported trainable_scope: %s" % trainable_scope)
    for module in trainable_modules:
        for param in module.parameters():
            param.requires_grad = True


class ECCPMimicCXRDataset(Dataset):
    def __init__(
        self,
        anno_path,
        data_dir,
        split,
        tokenizer,
        image_size=128,
        max_len=64,
        limit=-1,
        promptmrg_label_lookup=None,
        max_images=1,
    ):
        with open(anno_path, "r") as handle:
            annotation = json.load(handle)
        samples = annotation[split]
        if limit > 0:
            samples = samples[:limit]
        self.samples = samples
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_images = max_images
        self.promptmrg_label_lookup = promptmrg_label_lookup or {}
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def _encode_report(self, report):
        ids = self.tokenizer(report)[: self.max_len]
        if len(ids) < self.max_len:
            ids = ids + [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_paths = sample["image_path"][: self.max_images]
        if not image_paths:
            raise ValueError("Sample has no image_path: %s" % sample.get("id", idx))
        images = [
            self.transform(Image.open(self._resolve_image_path(relative_path)).convert("RGB"))
            for relative_path in image_paths
        ]
        if self.max_images == 1:
            image_tensor = images[0]
        else:
            while len(images) < self.max_images:
                images.append(images[-1].clone())
            image_tensor = torch.stack(images[: self.max_images])
        clinical = build_clinical_sample(sample["report"])
        if sample.get("id") in self.promptmrg_label_lookup:
            clinical = merge_promptmrg_labels(clinical, self.promptmrg_label_lookup[sample["id"]])
        return {
            "image": image_tensor,
            "report_ids": self._encode_report(sample["report"]),
            "entity_labels": torch.tensor(clinical["entity_labels"], dtype=torch.float32),
            "entity_label_mask": torch.tensor(clinical["entity_label_mask"], dtype=torch.float32),
            "plan_finding_prompt": torch.tensor(clinical["plan_finding_prompt"], dtype=torch.long),
            "anatomy_finding_labels": torch.tensor(clinical["anatomy_finding_labels"], dtype=torch.float32),
            "structured_plan": clinical["structured_plan"],
            "raw_report": sample["report"],
        }

    def _resolve_image_path(self, relative_path):
        candidates = [
            os.path.join(self.data_dir, relative_path),
            os.path.join(self.data_dir.replace("images", "images300"), relative_path),
            os.path.join(os.path.dirname(self.data_dir), "images300", relative_path),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError("Could not resolve image path from candidates: %s" % candidates)


def collate_fn(batch):
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "report_ids": torch.stack([item["report_ids"] for item in batch]),
        "entity_labels": torch.stack([item["entity_labels"] for item in batch]),
        "entity_label_mask": torch.stack([item["entity_label_mask"] for item in batch]),
        "plan_finding_prompt": torch.stack([item["plan_finding_prompt"] for item in batch]),
        "anatomy_finding_labels": torch.stack([item["anatomy_finding_labels"] for item in batch]),
        "structured_plan": [item["structured_plan"] for item in batch],
        "raw_report": [item["raw_report"] for item in batch],
    }


def _aggregate_promptmrg_states(current, incoming):
    if current == 1 or incoming == 1:
        return 1
    if current == 3 or incoming == 3:
        return 3
    if current == 2 or incoming == 2:
        return 2
    return 0


def load_promptmrg_label_lookup(promptmrg_anno_path):
    if not promptmrg_anno_path:
        return {}
    with open(promptmrg_anno_path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)

    lookup = {}
    for row in rows:
        sample_id = row.get("id")
        labels = row.get("labels")
        if sample_id is None or labels is None:
            continue
        if sample_id not in lookup:
            lookup[sample_id] = [0] * len(labels)
        lookup[sample_id] = [
            _aggregate_promptmrg_states(old, int(new))
            for old, new in zip(lookup[sample_id], labels)
        ]
    return lookup


def decode_reports(tokenizer, token_batch):
    reports = []
    for token_ids in token_batch:
        ids = [
            int(item)
            for item in token_ids
            if int(item) not in (0, 1, 2)
        ]
        reports.append(tokenizer.decode(ids))
    return reports


def postprocess_report(report):
    normalized = " ".join(report.replace("..", ".").split())
    if not normalized:
        return normalized
    raw_parts = normalized.split(".")
    sentences = []
    seen = set()
    for index, part in enumerate(raw_parts):
        sentence = " ".join(part.split())
        if not sentence:
            continue
        is_final_fragment = index == len(raw_parts) - 1 and not normalized.endswith(".")
        if is_final_fragment and sentences:
            continue
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        sentences.append(sentence)
    return " . ".join(sentences) + (" ." if sentences else "")


def _ngram_counts(tokens, n):
    return {
        tuple(tokens[i : i + n]): tokens[i : i + n].count(tokens[i])
        for i in range(max(len(tokens) - n + 1, 0))
    }


def simple_bleu(reference, prediction, max_n=4):
    ref_tokens = reference.split()
    pred_tokens = prediction.split()
    if not pred_tokens:
        return [0.0] * max_n
    scores = []
    for n in range(1, max_n + 1):
        pred_ngrams = {}
        for i in range(max(len(pred_tokens) - n + 1, 0)):
            ngram = tuple(pred_tokens[i : i + n])
            pred_ngrams[ngram] = pred_ngrams.get(ngram, 0) + 1
        ref_ngrams = {}
        for i in range(max(len(ref_tokens) - n + 1, 0)):
            ngram = tuple(ref_tokens[i : i + n])
            ref_ngrams[ngram] = ref_ngrams.get(ngram, 0) + 1
        overlap = sum(min(count, ref_ngrams.get(ngram, 0)) for ngram, count in pred_ngrams.items())
        total = max(sum(pred_ngrams.values()), 1)
        scores.append(overlap / total)
    return scores


def simple_rouge_l(reference, prediction):
    ref_tokens = reference.split()
    pred_tokens = prediction.split()
    if not ref_tokens or not pred_tokens:
        return 0.0
    dp = [[0] * (len(pred_tokens) + 1) for _ in range(len(ref_tokens) + 1)]
    for i, ref_token in enumerate(ref_tokens, 1):
        for j, pred_token in enumerate(pred_tokens, 1):
            if ref_token == pred_token:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def compute_lightweight_nlg(gts, res):
    bleu_totals = np.zeros(4, dtype=np.float64)
    rouge_scores = []
    for key in gts:
        reference = gts[key][0]
        prediction = res[key][0]
        bleu_totals += np.array(simple_bleu(reference, prediction), dtype=np.float64)
        rouge_scores.append(simple_rouge_l(reference, prediction))
    denom = max(len(gts), 1)
    return {
        "BLEU_1": float(bleu_totals[0] / denom),
        "BLEU_2": float(bleu_totals[1] / denom),
        "BLEU_3": float(bleu_totals[2] / denom),
        "BLEU_4": float(bleu_totals[3] / denom),
        "ROUGE_L": float(np.mean(rouge_scores)) if rouge_scores else 0.0,
    }


def compute_binary_prf(target, prediction):
    target = [int(x) for x in target]
    prediction = [int(x) for x in prediction]
    tp = sum(1 for y, y_hat in zip(target, prediction) if y == 1 and y_hat == 1)
    fp = sum(1 for y, y_hat in zip(target, prediction) if y == 0 and y_hat == 1)
    fn = sum(1 for y, y_hat in zip(target, prediction) if y == 1 and y_hat == 0)
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def compute_masked_entity_prf(target, prediction, mask):
    filtered_target = [int(y) for y, keep in zip(target, mask) if int(keep) == 1]
    filtered_prediction = [int(y_hat) for y_hat, keep in zip(prediction, mask) if int(keep) == 1]
    if not filtered_target:
        return 0.0, 0.0, 0.0
    return compute_binary_prf(filtered_target, filtered_prediction)


def compute_masked_entity_f1(target, prediction, mask):
    return compute_masked_entity_prf(target, prediction, mask)[2]


def compute_report_entity_prf(reference_report, prediction_report):
    reference = build_clinical_sample(reference_report)
    prediction = build_clinical_sample(prediction_report)
    mask = reference["entity_label_mask"]
    return compute_masked_entity_prf(reference["entity_labels"], prediction["entity_labels"], mask)


def compute_selection_score(metrics, metric_name):
    if metric_name == "composite":
        return (
            0.5 * float(metrics.get("ROUGE_L", 0.0))
            + 0.3 * float(metrics.get("Entity_F1", 0.0))
            + 0.2 * float(metrics.get("Plan_Report_Consistency", 0.0))
        )
    return float(metrics.get(metric_name, 0.0))


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    lambda_entity,
    lambda_relation,
    lambda_intervention,
    label_smoothing,
    log_interval=0,
):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    non_blocking = device.type == "cuda"
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=non_blocking)
        report_ids = batch["report_ids"].to(device, non_blocking=non_blocking)
        entity_labels = batch["entity_labels"].to(device, non_blocking=non_blocking)
        entity_label_mask = batch["entity_label_mask"].to(device, non_blocking=non_blocking)
        plan_finding_prompt = batch["plan_finding_prompt"].to(device, non_blocking=non_blocking)
        anatomy_finding_labels = batch["anatomy_finding_labels"].to(device, non_blocking=non_blocking)

        outputs = model(images, report_ids[:, :-1], plan_finding_prompt=plan_finding_prompt)
        gen_loss = F.cross_entropy(
            outputs["logits"].reshape(-1, outputs["logits"].size(-1)),
            report_ids[:, 1:].reshape(-1),
            ignore_index=0,
            label_smoothing=label_smoothing,
        )
        raw_entity_loss = F.binary_cross_entropy_with_logits(
            outputs["entity_logits"], entity_labels, reduction="none"
        )
        entity_loss = (raw_entity_loss * entity_label_mask).sum() / entity_label_mask.sum().clamp_min(1.0)
        relation_loss = F.binary_cross_entropy_with_logits(
            outputs["anatomy_finding_logits"], anatomy_finding_labels
        )
        intervention_loss = torch.relu(-outputs["interventional_effects"]).mean()
        loss = (
            gen_loss
            + lambda_entity * entity_loss
            + lambda_relation * relation_loss
            + lambda_intervention * intervention_loss
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item())
        if log_interval and step % log_interval == 0:
            elapsed = time.time() - start_time
            avg_loss = total_loss / step
            print(
                f"train step {step}/{len(loader)} avg_loss={avg_loss:.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model,
    loader,
    tokenizer,
    device,
    max_len,
    generation_plan_source="predicted",
    num_beams=1,
    repetition_penalty=1.0,
    length_penalty=1.0,
    no_repeat_ngram_size=0,
    postprocess_reports=False,
    log_interval=0,
    split_name="eval",
):
    model.eval()
    gts = {}
    res = {}
    entity_precision_scores = []
    entity_recall_scores = []
    entity_scores = []
    report_entity_precision_scores = []
    report_entity_recall_scores = []
    report_entity_scores = []
    consistency_rows = []
    prc_scores = []
    esr_scores = []
    ifs_scores = []
    examples = []
    start_time = time.time()
    non_blocking = device.type == "cuda"

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=non_blocking)
        target_plan_prompt = batch["plan_finding_prompt"].to(device, non_blocking=non_blocking)
        predicted_plan_prompt = model.predict_plan_finding_prompt(images)
        generation_prompt = target_plan_prompt if generation_plan_source == "target" else predicted_plan_prompt
        generated = model.generate(
            images,
            max_len=max_len,
            plan_finding_prompt=generation_prompt,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        generated_reports = decode_reports(tokenizer, generated.cpu().tolist())
        if postprocess_reports:
            generated_reports = [postprocess_report(report) for report in generated_reports]
        reference_reports = decode_reports(tokenizer, batch["report_ids"].cpu().tolist())
        outputs = model(
            images,
            batch["report_ids"].to(device, non_blocking=non_blocking),
            plan_finding_prompt=target_plan_prompt,
        )
        entity_pred = (torch.sigmoid(outputs["entity_logits"]) > 0.5).long().cpu().tolist()
        effects = outputs["interventional_effects"].detach().cpu()

        offset = len(gts)
        target_entity_masks = batch["entity_label_mask"].long().tolist()
        for idx, (pred, ref, plan, target_entities, target_mask) in enumerate(
            zip(
                generated_reports,
                reference_reports,
                batch["structured_plan"],
                batch["entity_labels"].long().tolist(),
                target_entity_masks,
            )
        ):
            gts[offset + idx] = [ref]
            res[offset + idx] = [pred]
            entity_precision, entity_recall, entity_f1 = compute_masked_entity_prf(
                target_entities, entity_pred[idx], target_mask
            )
            report_precision, report_recall, report_f1 = compute_report_entity_prf(ref, pred)
            entity_precision_scores.append(entity_precision)
            entity_recall_scores.append(entity_recall)
            entity_scores.append(entity_f1)
            report_entity_precision_scores.append(report_precision)
            report_entity_recall_scores.append(report_recall)
            report_entity_scores.append(report_f1)
            internal_plan = (
                plan_prompt_to_structured_plan(predicted_plan_prompt[idx].detach().cpu().tolist())
                if generation_plan_source == "predicted"
                else plan
            )
            consistency_rows.append({"reference": ref, "prediction": pred, "plan": internal_plan})
            prc_scores.append(compute_plan_report_consistency(internal_plan, pred))
            esr_scores.append(compute_evidence_support_rate(internal_plan, pred))
            ifs_scores.append(float((effects[idx] > 0.01).float().mean().item()))
            if len(examples) < 3:
                examples.append({"reference": ref, "prediction": pred, "plan": internal_plan[:3]})
        if log_interval and step % log_interval == 0:
            elapsed = time.time() - start_time
            print(f"{split_name} step {step}/{len(loader)} elapsed={elapsed:.1f}s", flush=True)

    nlg = {}
    if compute_scores is not None and gts:
        try:
            nlg = compute_scores(gts, res)
        except Exception as exc:
            nlg = compute_lightweight_nlg(gts, res)
            nlg["NLG_NOTE"] = "Used lightweight fallback because pycocoevalcap failed: %s" % exc
    elif gts:
        nlg = compute_lightweight_nlg(gts, res)
        nlg["NLG_NOTE"] = "Used lightweight fallback because pycocoevalcap is unavailable."

    consistency = compute_internal_consistency(consistency_rows)
    nlg.update(consistency)

    nlg.update(
        {
            "Entity_Precision": float(np.mean(entity_precision_scores)) if entity_precision_scores else 0.0,
            "Entity_Recall": float(np.mean(entity_recall_scores)) if entity_recall_scores else 0.0,
            "Entity_F1": float(np.mean(entity_scores)) if entity_scores else 0.0,
            "Report_Entity_Precision": float(np.mean(report_entity_precision_scores)) if report_entity_precision_scores else 0.0,
            "Report_Entity_Recall": float(np.mean(report_entity_recall_scores)) if report_entity_recall_scores else 0.0,
            "Report_Entity_F1": float(np.mean(report_entity_scores)) if report_entity_scores else 0.0,
            "Plan_Report_Consistency": float(np.mean(prc_scores)) if prc_scores else 0.0,
            "Evidence_Support_Rate": float(np.mean(esr_scores)) if esr_scores else 0.0,
            "Interventional_Finding_Sensitivity": float(np.mean(ifs_scores)) if ifs_scores else 0.0,
            "examples": examples,
        }
    )
    return nlg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_path", default="/root/autodl-tmp/ECCP/data/mimic_cxr/mimic_ammrg_annotation.json")
    parser.add_argument("--data_dir", default="/root/autodl-tmp/physionet.org/files/mimic-cxr-jpg/2.0.0/images300")
    parser.add_argument(
        "--promptmrg_anno_path",
        default="",
    )
    parser.add_argument("--output_dir", default="output/eccp_mimic")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--val_limit", type=int, default=None)
    parser.add_argument("--test_limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--max_images", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--gen_max_len", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--visual_backbone", choices=["simple", "resnet18", "resnet34"], default="simple")
    parser.add_argument("--pretrained_backbone", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--disable_view_embedding", action="store_true")
    parser.add_argument("--disable_plan_prompt", action="store_true")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval_split", choices=["val", "test"], default="test")
    parser.add_argument("--postprocess_reports", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--trainable_scope", choices=["all", "decoder"], default="all")
    parser.add_argument("--lambda_entity", type=float, default=0.5)
    parser.add_argument("--lambda_relation", type=float, default=0.3)
    parser.add_argument("--lambda_intervention", type=float, default=0.05)
    parser.add_argument("--best_metric", default="Plan_Report_Consistency")
    parser.add_argument("--generation_plan_source", choices=["predicted", "target"], default="predicted")
    parser.add_argument("--use_promptmrg_labels", action="store_true")
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--eval_log_interval", type=int, default=50)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--lr_scheduler_patience", type=int, default=0)
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=0.0)
    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    generation_max_len = args.max_len if args.gen_max_len is None else args.gen_max_len
    tokenizer = Tokenizer(ann_path=args.anno_path, threshold=10, dataset_name="mimic_cxr")
    vocab_size = max(tokenizer.idx2token.keys()) + 1

    train_limit = args.limit if args.train_limit is None else args.train_limit
    if args.val_limit is None:
        val_limit = max(1, args.limit // 2) if args.limit > 0 else -1
    else:
        val_limit = args.val_limit
    if args.test_limit is None:
        test_limit = max(1, args.limit // 2) if args.limit > 0 else -1
    else:
        test_limit = args.test_limit
    promptmrg_label_lookup = (
        load_promptmrg_label_lookup(args.promptmrg_anno_path)
        if args.use_promptmrg_labels
        else {}
    )

    train_dataset = ECCPMimicCXRDataset(
        args.anno_path,
        args.data_dir,
        "train",
        tokenizer,
        args.image_size,
        args.max_len,
        train_limit,
        promptmrg_label_lookup,
        args.max_images,
    )
    val_dataset = ECCPMimicCXRDataset(
        args.anno_path,
        args.data_dir,
        "val",
        tokenizer,
        args.image_size,
        args.max_len,
        val_limit,
        promptmrg_label_lookup,
        args.max_images,
    )
    test_dataset = ECCPMimicCXRDataset(
        args.anno_path,
        args.data_dir,
        "test",
        tokenizer,
        args.image_size,
        args.max_len,
        test_limit,
        promptmrg_label_lookup,
        args.max_images,
    )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory and device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": args.prefetch_factor,
            }
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    model = ECCPNet(
        vocab_size=vocab_size,
        num_anatomy_regions=len(ANATOMY_NAMES),
        num_findings=len(FINDING_NAMES),
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        max_report_len=args.max_len,
        visual_backbone=args.visual_backbone,
        pretrained_backbone=args.pretrained_backbone,
        freeze_backbone=args.freeze_backbone,
        max_views=max(args.max_images, 1),
        use_view_embedding=not args.disable_view_embedding,
        use_plan_prompt=not args.disable_plan_prompt,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "metrics.jsonl"
    best_score = None
    best_epoch = -1
    best_state = None

    print(
        f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | "
        f"Test samples: {len(test_dataset)} | Vocab: {vocab_size}"
    )
    print(
        f"Device: {device} | num_workers={args.num_workers} | "
        f"pin_memory={loader_kwargs['pin_memory']} | gen_max_len={generation_max_len}"
    )
    if args.use_promptmrg_labels:
        print(f"PromptMRG label cases loaded: {len(promptmrg_label_lookup)}")

    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
        if missing_keys or unexpected_keys:
            print(f"Checkpoint load with strict=False | missing={missing_keys} | unexpected={unexpected_keys}")
        print(f"Loaded checkpoint: {args.checkpoint_path}")

    configure_trainable_scope(model, args.trainable_scope)
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    if not args.eval_only:
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=args.lr,
            weight_decay=1e-4,
        )
        scheduler = None
        if args.lr_scheduler_patience > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=args.lr_scheduler_factor,
                patience=args.lr_scheduler_patience,
                min_lr=args.min_lr,
            )
        print(
            f"Trainable scope: {args.trainable_scope} | trainable_params={trainable_params} | "
            f"seed={args.seed} | label_smoothing={args.label_smoothing} | lr={args.lr}"
        )
    else:
        scheduler = None

    if args.eval_only:
        eval_loader = val_loader if args.eval_split == "val" else test_loader
        eval_metrics = evaluate(
            model,
            eval_loader,
            tokenizer,
            device,
            generation_max_len,
            args.generation_plan_source,
            args.num_beams,
            args.repetition_penalty,
            args.length_penalty,
            args.no_repeat_ngram_size,
            args.postprocess_reports,
            args.eval_log_interval,
            args.eval_split,
        )
        eval_metrics["Selection_Score"] = compute_selection_score(eval_metrics, args.best_metric)
        eval_path = output_dir / "eval_metrics.json"
        with open(eval_path, "w", encoding="utf-8") as eval_file:
            json.dump({"split": args.eval_split, **eval_metrics}, eval_file, ensure_ascii=False, indent=2)
        print(f"eval_only_{args.eval_split}_result:")
        print(json.dumps(eval_metrics, ensure_ascii=False, indent=2))
        print(f"Eval metrics: {eval_path}")
        return

    ckpt_path = output_dir / "eccp_mimic_best.pt"
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.lambda_entity,
            args.lambda_relation,
            args.lambda_intervention,
            args.label_smoothing,
            args.log_interval,
        )
        metrics = evaluate(
            model,
            val_loader,
            tokenizer,
            device,
            generation_max_len,
            args.generation_plan_source,
            args.num_beams,
            args.repetition_penalty,
            args.length_penalty,
            args.no_repeat_ngram_size,
            args.postprocess_reports,
            args.eval_log_interval,
            "val",
        )
        metric_value = compute_selection_score(metrics, args.best_metric)
        metrics["Selection_Score"] = metric_value
        improved = best_score is None or metric_value > best_score + args.early_stop_min_delta
        if improved:
            best_score = metric_value
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": best_state,
                    "args": vars(args),
                    "best_epoch": best_epoch,
                    "best_metric": args.best_metric,
                    "best_score": best_score,
                    "val_metrics": metrics,
                },
                ckpt_path,
            )
        else:
            epochs_without_improvement += 1
        if scheduler is not None:
            scheduler.step(metric_value)
        metrics["lr"] = optimizer.param_groups[0]["lr"]
        metrics["epochs_without_improvement"] = epochs_without_improvement
        log_record = {"epoch": epoch, "loss": loss, "split": "val", **metrics}
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(log_record, ensure_ascii=False) + "\n")
        print(
            f"Epoch {epoch}: loss={loss:.4f} | lr={optimizer.param_groups[0]['lr']:.6g} | "
            f"best_epoch={best_epoch} | no_improve={epochs_without_improvement}"
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch}: no improvement for "
                f"{epochs_without_improvement} epoch(s)."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(
        model,
        test_loader,
        tokenizer,
        device,
        generation_max_len,
        args.generation_plan_source,
        args.num_beams,
        args.repetition_penalty,
        args.length_penalty,
        args.no_repeat_ngram_size,
        args.postprocess_reports,
        args.eval_log_interval,
        "test",
    )
    test_metrics["Selection_Score"] = compute_selection_score(test_metrics, args.best_metric)
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps({"epoch": best_epoch, "split": "test", **test_metrics}, ensure_ascii=False) + "\n")

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "best_epoch": best_epoch,
            "best_metric": args.best_metric,
            "best_score": best_score,
            "test_metrics": test_metrics,
        },
        ckpt_path,
    )
    print(f"Best epoch: {best_epoch} | {args.best_metric}={best_score}")
    print("test_result:")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))
    print(f"Saved single checkpoint: {ckpt_path}")
    print(f"Metrics log: {log_path}")


if __name__ == "__main__":
    main()
