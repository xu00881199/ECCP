import argparse
import copy
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from datasets.ffair_plan_builder import (
    FFAIR_ANATOMY_NAMES,
    FFAIR_FINDING_NAMES,
    build_ffair_clinical_sample,
    compute_entity_f1,
    compute_ffair_evidence_support_rate,
    compute_ffair_plan_report_consistency,
)
from datasets.tokenizers import Tokenizer
from models.acrp import ECCPNet

try:
    from utils.engine import compute_scores
except Exception:
    compute_scores = None


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


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


def split_ffair_reports(rows, seed=42, train_ratio=0.8, val_ratio=0.1):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    if len(rows) < 3:
        return rows, [], []
    train_len = int(len(rows) * train_ratio)
    val_len = int(len(rows) * val_ratio)
    train_len = max(train_len, 1)
    val_len = max(val_len, 1)
    if train_len + val_len >= len(rows):
        val_len = max(0, len(rows) - train_len)
    return rows[:train_len], rows[train_len : train_len + val_len], rows[train_len + val_len :]


def _is_valid_report_text(value):
    if not isinstance(value, str):
        return False
    return bool(value.strip())


def filter_valid_ffair_reports(rows):
    return [
        row
        for row in rows
        if row.get("Path") and _is_valid_report_text(row.get("Finding-English") or row.get("report"))
    ]


def load_ffair_lesion_lookup(lesion_info_path=None, class_mapping_path=None):
    if not lesion_info_path or not class_mapping_path:
        return {}, 0
    with open(lesion_info_path, "r", encoding="utf-8") as handle:
        lesion_info = json.load(handle)
    with open(class_mapping_path, "r", encoding="utf-8") as handle:
        class_mapping = json.load(handle)

    num_classes = max(int(idx) for idx in class_mapping.values()) + 1 if class_mapping else 0
    lookup = {}
    for patient_id, image_items in lesion_info.items():
        labels = [0.0] * num_classes
        for lesions in image_items.values():
            for lesion in lesions:
                if not lesion:
                    continue
                class_id = str(lesion[0])
                if class_id in class_mapping:
                    labels[int(class_mapping[class_id])] = 1.0
        lookup[patient_id] = labels
    return lookup, num_classes


def _image_sort_key(filename):
    match = re.search(r"(\d+)", filename)
    return (int(match.group(1)) if match else 10**9, filename)


def list_patient_images(data_dir, patient_path):
    patient_dir = os.path.join(data_dir, patient_path)
    if not os.path.isdir(patient_dir):
        raise FileNotFoundError("Missing FFA-IR patient folder: %s" % patient_dir)
    images = []
    for dirpath, dirnames, filenames in os.walk(patient_dir):
        dirnames[:] = [name for name in dirnames if name != ".ipynb_checkpoints"]
        if ".ipynb_checkpoints" in dirpath:
            continue
        for filename in filenames:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                full_path = os.path.join(dirpath, filename)
                images.append(os.path.relpath(full_path, data_dir).replace("\\", "/"))
    return sorted(images, key=lambda path: _image_sort_key(os.path.basename(path)))


def sample_temporal_images(image_paths, max_images):
    if not image_paths:
        raise ValueError("Cannot sample an empty image list.")
    if len(image_paths) >= max_images:
        indices = np.linspace(0, len(image_paths) - 1, max_images, dtype=int)
        return [image_paths[int(idx)] for idx in indices]
    sampled = []
    while len(sampled) < max_images:
        sampled.extend(image_paths)
    return sampled[:max_images]


def sample_phase_balanced_images(image_paths, max_images):
    if not image_paths:
        raise ValueError("Cannot sample an empty image list.")
    if len(image_paths) < max_images:
        sampled = []
        while len(sampled) < max_images:
            sampled.extend(image_paths)
        image_paths = sampled[:max_images]
    n_images = len(image_paths)
    phase_bounds = [
        (0, max(1, int(np.ceil(n_images * 0.25)))),
        (max(1, int(np.ceil(n_images * 0.25))), max(2, int(np.ceil(n_images * 0.65)))),
        (max(2, int(np.ceil(n_images * 0.65))), n_images),
    ]
    base_quota = [max_images // 3] * 3
    for idx in range(max_images % 3):
        base_quota[idx if idx == 0 else 3 - idx] += 1

    chosen = []
    phase_ids = []
    for phase_id, ((start, end), quota) in enumerate(zip(phase_bounds, base_quota)):
        segment = list(range(start, max(start + 1, end)))
        segment = [min(max(index, 0), n_images - 1) for index in segment]
        if len(segment) >= quota:
            indices = np.linspace(0, len(segment) - 1, quota, dtype=int)
            sampled = [segment[int(index)] for index in indices]
        else:
            sampled = []
            while len(sampled) < quota:
                sampled.extend(segment)
            sampled = sampled[:quota]
        chosen.extend(sampled)
        phase_ids.extend([phase_id] * len(sampled))

    paired = sorted(zip(chosen, phase_ids), key=lambda item: item[0])
    sampled_paths = [image_paths[index] for index, _ in paired[:max_images]]
    sampled_phase_ids = [phase_id for _, phase_id in paired[:max_images]]
    return sampled_paths, sampled_phase_ids


def sample_ffair_images(image_paths, max_images, sampling_strategy):
    if sampling_strategy == "phase_balanced":
        return sample_phase_balanced_images(image_paths, max_images)
    sampled = sample_temporal_images(image_paths, max_images)
    phase_ids = []
    for idx in range(len(sampled)):
        relative_position = 0.0 if len(sampled) == 1 else idx / float(len(sampled) - 1)
        if relative_position < 0.25:
            phase_ids.append(0)
        elif relative_position < 0.65:
            phase_ids.append(1)
        else:
            phase_ids.append(2)
    return sampled, phase_ids


class ECCPFFAIRDataset(Dataset):
    def __init__(
        self,
        samples,
        data_dir,
        tokenizer,
        image_size=160,
        max_len=128,
        max_images=8,
        lesion_lookup=None,
        num_lesion_classes=0,
        sampling_strategy="uniform",
    ):
        self.samples = list(samples)
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_images = max_images
        self.lesion_lookup = lesion_lookup or {}
        self.num_lesion_classes = num_lesion_classes
        self.sampling_strategy = sampling_strategy
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
        if self.tokenizer is None:
            ids = [1, 2]
        else:
            ids = self.tokenizer(report)[: self.max_len]
        if len(ids) < self.max_len:
            ids = ids + [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        report = sample.get("Finding-English") or sample.get("report") or ""
        patient_path = sample["Path"]
        all_images = list_patient_images(self.data_dir, patient_path)
        image_paths, phase_ids = sample_ffair_images(all_images, self.max_images, self.sampling_strategy)
        images = [
            self.transform(Image.open(os.path.join(self.data_dir, relative_path)).convert("RGB"))
            for relative_path in image_paths
        ]
        clinical = build_ffair_clinical_sample(report)
        lesion_labels = self.lesion_lookup.get(patient_path)
        if lesion_labels is None:
            lesion_labels = [0.0] * self.num_lesion_classes
            lesion_label_mask = 0.0
        else:
            lesion_label_mask = 1.0
        return {
            "image": torch.stack(images),
            "view_phase_ids": torch.tensor(phase_ids, dtype=torch.long),
            "report_ids": self._encode_report(report),
            "entity_labels": torch.tensor(clinical["entity_labels"], dtype=torch.float32),
            "entity_label_mask": torch.tensor(clinical["entity_label_mask"], dtype=torch.float32),
            "plan_finding_prompt": torch.tensor(clinical["plan_finding_prompt"], dtype=torch.long),
            "anatomy_finding_labels": torch.tensor(clinical["anatomy_finding_labels"], dtype=torch.float32),
            "structured_plan": clinical["structured_plan"],
            "raw_report": report,
            "image_paths": image_paths,
            "case_id": sample.get("SID", patient_path),
            "lesion_labels": torch.tensor(lesion_labels, dtype=torch.float32),
            "lesion_label_mask": torch.tensor(lesion_label_mask, dtype=torch.float32),
        }


def collate_fn(batch):
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "view_phase_ids": torch.stack([item["view_phase_ids"] for item in batch]),
        "report_ids": torch.stack([item["report_ids"] for item in batch]),
        "entity_labels": torch.stack([item["entity_labels"] for item in batch]),
        "entity_label_mask": torch.stack([item["entity_label_mask"] for item in batch]),
        "plan_finding_prompt": torch.stack([item["plan_finding_prompt"] for item in batch]),
        "anatomy_finding_labels": torch.stack([item["anatomy_finding_labels"] for item in batch]),
        "structured_plan": [item["structured_plan"] for item in batch],
        "raw_report": [item["raw_report"] for item in batch],
        "image_paths": [item["image_paths"] for item in batch],
        "case_id": [item["case_id"] for item in batch],
        "lesion_labels": torch.stack([item["lesion_labels"] for item in batch]),
        "lesion_label_mask": torch.stack([item["lesion_label_mask"] for item in batch]),
    }


def decode_reports(tokenizer, token_batch):
    reports = []
    for token_ids in token_batch:
        ids = [int(item) for item in token_ids if int(item) not in (0, 1, 2)]
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
        scores.append(overlap / max(sum(pred_ngrams.values()), 1))
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
        bleu_totals += np.array(simple_bleu(gts[key][0], res[key][0]), dtype=np.float64)
        rouge_scores.append(simple_rouge_l(gts[key][0], res[key][0]))
    denom = max(len(gts), 1)
    return {
        "BLEU_1": float(bleu_totals[0] / denom),
        "BLEU_2": float(bleu_totals[1] / denom),
        "BLEU_3": float(bleu_totals[2] / denom),
        "BLEU_4": float(bleu_totals[3] / denom),
        "ROUGE_L": float(np.mean(rouge_scores)) if rouge_scores else 0.0,
    }


def compute_masked_entity_f1(target, prediction, mask):
    filtered_target = [int(y) for y, keep in zip(target, mask) if int(keep) == 1]
    filtered_prediction = [int(y_hat) for y_hat, keep in zip(prediction, mask) if int(keep) == 1]
    if not filtered_target:
        return 0.0
    return compute_entity_f1(filtered_target, filtered_prediction)


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
    lambda_lesion,
    label_smoothing,
):
    model.train()
    total_loss = 0.0
    non_blocking = device.type == "cuda"
    for batch in loader:
        images = batch["image"].to(device, non_blocking=non_blocking)
        view_phase_ids = batch["view_phase_ids"].to(device, non_blocking=non_blocking)
        report_ids = batch["report_ids"].to(device, non_blocking=non_blocking)
        entity_labels = batch["entity_labels"].to(device, non_blocking=non_blocking)
        entity_label_mask = batch["entity_label_mask"].to(device, non_blocking=non_blocking)
        plan_finding_prompt = batch["plan_finding_prompt"].to(device, non_blocking=non_blocking)
        anatomy_finding_labels = batch["anatomy_finding_labels"].to(device, non_blocking=non_blocking)
        lesion_labels = batch["lesion_labels"].to(device, non_blocking=non_blocking)
        lesion_label_mask = batch["lesion_label_mask"].to(device, non_blocking=non_blocking)

        outputs = model(
            images,
            report_ids[:, :-1],
            plan_finding_prompt=plan_finding_prompt,
            view_phase_ids=view_phase_ids,
        )
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
        lesion_loss = torch.tensor(0.0, device=device)
        if lambda_lesion > 0 and "lesion_logits" in outputs and lesion_labels.numel() > 0:
            raw_lesion_loss = F.binary_cross_entropy_with_logits(
                outputs["lesion_logits"], lesion_labels, reduction="none"
            ).mean(dim=1)
            lesion_loss = (raw_lesion_loss * lesion_label_mask).sum() / lesion_label_mask.sum().clamp_min(1.0)
        loss = (
            gen_loss
            + lambda_entity * entity_loss
            + lambda_relation * relation_loss
            + lambda_intervention * intervention_loss
            + lambda_lesion * lesion_loss
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def train_language_prior_epoch(
    model,
    loader,
    optimizer,
    device,
    label_smoothing,
):
    model.train()
    total_loss = 0.0
    non_blocking = device.type == "cuda"
    for batch in loader:
        report_ids = batch["report_ids"].to(device, non_blocking=non_blocking)
        outputs = model.forward_language_model(report_ids[:, :-1])
        loss = F.cross_entropy(
            outputs["logits"].reshape(-1, outputs["logits"].size(-1)),
            report_ids[:, 1:].reshape(-1),
            ignore_index=0,
            label_smoothing=label_smoothing,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item())
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
):
    model.eval()
    gts = {}
    res = {}
    entity_scores = []
    prc_scores = []
    esr_scores = []
    examples = []
    non_blocking = device.type == "cuda"

    for batch in loader:
        images = batch["image"].to(device, non_blocking=non_blocking)
        view_phase_ids = batch["view_phase_ids"].to(device, non_blocking=non_blocking)
        target_plan_prompt = batch["plan_finding_prompt"].to(device, non_blocking=non_blocking)
        generation_prompt = target_plan_prompt if generation_plan_source == "target" else None
        generated = model.generate(
            images,
            max_len=max_len,
            plan_finding_prompt=generation_prompt,
            view_phase_ids=view_phase_ids,
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
            view_phase_ids=view_phase_ids,
        )
        entity_pred = (torch.sigmoid(outputs["entity_logits"]) > 0.5).long().cpu().tolist()
        target_entity_masks = batch["entity_label_mask"].long().tolist()
        offset = len(gts)
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
            entity_scores.append(compute_masked_entity_f1(target_entities, entity_pred[idx], target_mask))
            prc_scores.append(compute_ffair_plan_report_consistency(plan, pred))
            esr_scores.append(compute_ffair_evidence_support_rate(plan, pred))
            if len(examples) < 3:
                examples.append(
                    {
                        "case_id": batch["case_id"][idx],
                        "image_paths": batch["image_paths"][idx],
                        "reference": ref,
                        "prediction": pred,
                        "plan": plan[:3],
                    }
                )

    if compute_scores is not None and gts:
        try:
            nlg = compute_scores(gts, res)
        except Exception as exc:
            nlg = compute_lightweight_nlg(gts, res)
            nlg["NLG_NOTE"] = "Used lightweight fallback because pycocoevalcap failed: %s" % exc
    elif gts:
        nlg = compute_lightweight_nlg(gts, res)
        nlg["NLG_NOTE"] = "Used lightweight fallback because pycocoevalcap is unavailable."
    else:
        nlg = compute_lightweight_nlg(gts, res)

    nlg.update(
        {
            "Entity_F1": float(np.mean(entity_scores)) if entity_scores else 0.0,
            "Plan_Report_Consistency": float(np.mean(prc_scores)) if prc_scores else 0.0,
            "Evidence_Support_Rate": float(np.mean(esr_scores)) if esr_scores else 0.0,
            "examples": examples,
        }
    )
    return nlg


def load_reports(anno_path):
    with open(anno_path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError("FFA-IR report annotation must be a list: %s" % anno_path)
    filtered = filter_valid_ffair_reports(rows)
    skipped = len(rows) - len(filtered)
    if skipped:
        print(f"Skipped {skipped} FFA-IR rows with missing/non-string reports or paths.")
    return filtered


def maybe_limit(rows, limit):
    if limit is not None and limit > 0:
        return rows[:limit]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_path", default="/root/autodl-tmp/ECCP/data/ffa-ir/1.1.0/report.json")
    parser.add_argument("--data_dir", default="/root/autodl-tmp/ECCP/data/ffa-ir/1.1.0/FFAIR_1")
    parser.add_argument("--lesion_info_path", default=None)
    parser.add_argument("--class_mapping_path", default=None)
    parser.add_argument("--output_dir", default="output/eccp_ffair")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--val_limit", type=int, default=None)
    parser.add_argument("--test_limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--image_size", type=int, default=160)
    parser.add_argument("--max_images", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--gen_max_len", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--visual_backbone", choices=["simple", "resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--pretrained_backbone", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--disable_view_embedding", action="store_true")
    parser.add_argument("--use_phase_embedding", action="store_true")
    parser.add_argument("--sampling_strategy", choices=["uniform", "phase_balanced"], default="uniform")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--trainable_scope", choices=["all", "decoder"], default="all")
    parser.add_argument("--lambda_entity", type=float, default=0.5)
    parser.add_argument("--lambda_relation", type=float, default=0.3)
    parser.add_argument("--lambda_intervention", type=float, default=0.05)
    parser.add_argument("--lambda_lesion", type=float, default=0.0)
    parser.add_argument("--use_lesion_prompt", action="store_true")
    parser.add_argument("--num_language_prior_tokens", type=int, default=0)
    parser.add_argument("--lm_warmup_epochs", type=int, default=0)
    parser.add_argument("--lm_warmup_lr", type=float, default=None)
    parser.add_argument("--best_metric", default="ROUGE_L")
    parser.add_argument("--generation_plan_source", choices=["predicted", "target"], default="predicted")
    parser.add_argument("--tokenizer_dataset_name", default="ffa_ir")
    parser.add_argument("--vocab_threshold", type=int, default=3)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    generation_max_len = args.max_len if args.gen_max_len is None else args.gen_max_len

    rows = load_reports(args.anno_path)
    train_rows, val_rows, test_rows = split_ffair_reports(rows, seed=args.seed)
    train_limit = args.limit if args.train_limit is None else args.train_limit
    val_limit = args.limit if args.val_limit is None else args.val_limit
    test_limit = args.limit if args.test_limit is None else args.test_limit
    train_rows = maybe_limit(train_rows, train_limit)
    val_rows = maybe_limit(val_rows, val_limit)
    test_rows = maybe_limit(test_rows, test_limit)

    tokenizer = Tokenizer(
        ann_path=args.anno_path,
        threshold=args.vocab_threshold,
        dataset_name=args.tokenizer_dataset_name,
    )
    vocab_size = max(tokenizer.idx2token.keys()) + 1
    lesion_lookup, num_lesion_classes = load_ffair_lesion_lookup(
        args.lesion_info_path, args.class_mapping_path
    )
    if num_lesion_classes:
        print(f"Lesion weak supervision cases: {len(lesion_lookup)} | classes: {num_lesion_classes}")

    train_dataset = ECCPFFAIRDataset(
        train_rows,
        args.data_dir,
        tokenizer,
        args.image_size,
        args.max_len,
        args.max_images,
        lesion_lookup,
        num_lesion_classes,
        args.sampling_strategy,
    )
    val_dataset = ECCPFFAIRDataset(
        val_rows,
        args.data_dir,
        tokenizer,
        args.image_size,
        args.max_len,
        args.max_images,
        lesion_lookup,
        num_lesion_classes,
        args.sampling_strategy,
    )
    test_dataset = ECCPFFAIRDataset(
        test_rows,
        args.data_dir,
        tokenizer,
        args.image_size,
        args.max_len,
        args.max_images,
        lesion_lookup,
        num_lesion_classes,
        args.sampling_strategy,
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
        num_anatomy_regions=len(FFAIR_ANATOMY_NAMES),
        num_findings=len(FFAIR_FINDING_NAMES),
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        max_report_len=args.max_len,
        visual_backbone=args.visual_backbone,
        pretrained_backbone=args.pretrained_backbone,
        freeze_backbone=args.freeze_backbone,
        max_views=max(args.max_images, 1),
        use_view_embedding=not args.disable_view_embedding,
        use_phase_embedding=args.use_phase_embedding,
        use_plan_prompt=not args.disable_plan_prompt,
        num_lesion_classes=num_lesion_classes if args.lambda_lesion > 0 else 0,
        use_lesion_prompt=args.use_lesion_prompt,
        num_language_prior_tokens=args.num_language_prior_tokens,
    ).to(device)

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
        f"Device: {device} | image_size={args.image_size} | max_images={args.max_images} | "
        f"num_workers={args.num_workers} | gen_max_len={generation_max_len}"
    )

    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
        if missing_keys or unexpected_keys:
            print(f"Checkpoint load with strict=False | missing={missing_keys} | unexpected={unexpected_keys}")
        print(f"Loaded checkpoint: {args.checkpoint_path}")

    configure_trainable_scope(model, args.trainable_scope)
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

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
        )
        eval_metrics["Selection_Score"] = compute_selection_score(eval_metrics, args.best_metric)
        eval_path = output_dir / "eval_metrics.json"
        with open(eval_path, "w", encoding="utf-8") as eval_file:
            json.dump({"split": args.eval_split, **eval_metrics}, eval_file, ensure_ascii=False, indent=2)
        print(f"eval_only_{args.eval_split}_result:")
        print(json.dumps(eval_metrics, ensure_ascii=False, indent=2))
        print(f"Eval metrics: {eval_path}")
        return

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    if args.lm_warmup_epochs > 0:
        warmup_lr = args.lr if args.lm_warmup_lr is None else args.lm_warmup_lr
        warmup_optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=warmup_lr,
            weight_decay=1e-4,
        )
        for warmup_epoch in range(args.lm_warmup_epochs):
            warmup_loss = train_language_prior_epoch(
                model,
                train_loader,
                warmup_optimizer,
                device,
                args.label_smoothing,
            )
            print(f"LM warmup epoch {warmup_epoch}: loss={warmup_loss:.4f}")
    print(
        f"Trainable scope: {args.trainable_scope} | trainable_params={trainable_params} | "
        f"seed={args.seed} | label_smoothing={args.label_smoothing} | "
        f"language_prior_tokens={args.num_language_prior_tokens}"
    )

    for epoch in range(args.epochs):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.lambda_entity,
            args.lambda_relation,
            args.lambda_intervention,
            args.lambda_lesion,
            args.label_smoothing,
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
        )
        metric_value = compute_selection_score(metrics, args.best_metric)
        metrics["Selection_Score"] = metric_value
        if best_score is None or metric_value > best_score:
            best_score = metric_value
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        log_record = {"epoch": epoch, "loss": loss, "split": "val", **metrics}
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(log_record, ensure_ascii=False) + "\n")
        print(f"Epoch {epoch}: loss={loss:.4f}")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

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
    )
    test_metrics["Selection_Score"] = compute_selection_score(test_metrics, args.best_metric)
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps({"epoch": best_epoch, "split": "test", **test_metrics}, ensure_ascii=False) + "\n")

    ckpt_path = output_dir / "eccp_ffair_best.pt"
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
