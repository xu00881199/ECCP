import re
from typing import Dict, Iterable, List, Sequence


FINDING_RULES = {
    "opacity": ["opacity", "opacities", "infiltrate", "infiltration"],
    "consolidation": ["consolidation", "consolidative"],
    "atelectasis": ["atelectasis", "atelectatic"],
    "pleural_effusion": ["pleural effusion", "effusion", "pleural fluid"],
    "pneumothorax": ["pneumothorax"],
    "cardiomegaly": ["cardiomegaly", "enlarged heart", "cardiac enlargement"],
    "edema": ["edema", "vascular congestion", "pulmonary vascular congestion"],
    "nodule": ["nodule", "nodular"],
    "mass": ["mass"],
    "fracture": ["fracture"],
}

ANATOMY_RULES = {
    "left_lung": ["left lung", "left hemithorax", "left chest"],
    "right_lung": ["right lung", "right hemithorax", "right chest"],
    "lung_bases": ["base", "bases", "basilar", "lower lung", "lower lobe"],
    "upper_lung": ["upper lung", "upper lobe", "apex", "apical"],
    "pleura": ["pleura", "pleural"],
    "heart": ["heart", "cardiac", "cardiomediastinal", "cardiomediastinum"],
    "mediastinum": ["mediastinum", "hilar", "hilum"],
}

NEGATIVE_PREFIXES = (
    "no",
    "without",
    "negative for",
    "no evidence of",
    "absence of",
    "free of",
)

UNCERTAIN_PREFIXES = (
    "possible",
    "possibly",
    "may represent",
    "cannot exclude",
    "questionable",
    "suggesting",
)

FINDING_NAMES = list(FINDING_RULES.keys())
ANATOMY_NAMES = list(ANATOMY_RULES.keys())
POLARITY_TO_ID = {"negative": 0, "positive": 1, "uncertain": 2}
ID_TO_POLARITY = {value: key for key, value in POLARITY_TO_ID.items()}

PROMPTMRG_CONDITIONS = [
    "enlarged cardiomediastinum",
    "cardiomegaly",
    "lung opacity",
    "lung lesion",
    "edema",
    "consolidation",
    "pneumonia",
    "atelectasis",
    "pneumothorax",
    "pleural effusion",
    "pleural other",
    "fracture",
    "support devices",
    "no finding",
]

PROMPTMRG_FINDING_MAP = {
    "cardiomegaly": "cardiomegaly",
    "lung opacity": "opacity",
    "edema": "edema",
    "consolidation": "consolidation",
    "atelectasis": "atelectasis",
    "pneumothorax": "pneumothorax",
    "pleural effusion": "pleural_effusion",
    "fracture": "fracture",
}


def normalize_report(text: str) -> str:
    text = text.lower().replace("\n", " ")
    text = re.sub(r"[^a-z0-9\s.]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains_any(text: str, phrases: Iterable[str]) -> bool:
    return any(re.search(r"\b" + re.escape(phrase) + r"\b", text) for phrase in phrases)


def _finding_polarity(text: str, finding_terms: Sequence[str]) -> str:
    for term in finding_terms:
        escaped = re.escape(term)
        for prefix in NEGATIVE_PREFIXES:
            if re.search(r"\b" + re.escape(prefix) + r"\s+(?:\w+\s+){0,8}" + escaped + r"\b", text):
                return "negative"
        for prefix in UNCERTAIN_PREFIXES:
            if re.search(r"\b" + re.escape(prefix) + r"\s+(?:\w+\s+){0,8}" + escaped + r"\b", text):
                return "uncertain"
    return "positive"


def extract_structured_plan(report: str) -> List[Dict[str, object]]:
    text = normalize_report(report)
    plan = []
    matched_anatomies = [
        anatomy for anatomy, terms in ANATOMY_RULES.items() if _contains_any(text, terms)
    ]
    if not matched_anatomies:
        matched_anatomies = ["left_lung", "right_lung"]

    for finding, terms in FINDING_RULES.items():
        if not _contains_any(text, terms):
            continue
        polarity = _finding_polarity(text, terms)
        if finding in ("pleural_effusion", "pneumothorax") and "pleura" not in matched_anatomies:
            anatomies = ["pleura"]
        elif finding == "cardiomegaly" and "heart" not in matched_anatomies:
            anatomies = ["heart"]
        else:
            anatomies = matched_anatomies
        for anatomy in anatomies:
            plan.append(
                {
                    "anatomy": anatomy,
                    "finding": finding,
                    "attribute": "none",
                    "polarity": polarity,
                    "evidence_score": 1.0 if polarity == "positive" else 0.5,
                }
            )
    return plan


def plan_to_tokens(plan: Sequence[Dict[str, object]]) -> List[str]:
    tokens = ["[BOS_PLAN]"]
    for item in plan:
        tokens.extend(
            [
                "[ANAT_%s]" % str(item["anatomy"]).upper(),
                "[FINDING_%s]" % str(item["finding"]).upper(),
                "[POL_%s]" % str(item["polarity"]).upper()[:3],
            ]
        )
    tokens.append("[EOS_PLAN]")
    return tokens


def build_clinical_sample(report: str) -> Dict[str, object]:
    plan = extract_structured_plan(report)
    entity_labels = [0] * len(FINDING_NAMES)
    anatomy_finding_labels = [[0] * len(FINDING_NAMES) for _ in ANATOMY_NAMES]
    polarity_labels = [POLARITY_TO_ID["negative"]] * len(FINDING_NAMES)

    for item in plan:
        finding_id = FINDING_NAMES.index(str(item["finding"]))
        anatomy_id = ANATOMY_NAMES.index(str(item["anatomy"]))
        polarity = str(item["polarity"])
        if polarity != "negative":
            entity_labels[finding_id] = 1
        anatomy_finding_labels[anatomy_id][finding_id] = 1
        polarity_labels[finding_id] = POLARITY_TO_ID[polarity]

    return {
        "structured_plan": plan,
        "plan_tokens": plan_to_tokens(plan),
        "entity_labels": entity_labels,
        "plan_finding_prompt": list(entity_labels),
        "entity_label_mask": [1] * len(FINDING_NAMES),
        "anatomy_finding_labels": anatomy_finding_labels,
        "polarity_labels": polarity_labels,
    }


def merge_promptmrg_labels(sample: Dict[str, object], promptmrg_labels: Sequence[int]) -> Dict[str, object]:
    merged = dict(sample)
    entity_labels = list(merged["entity_labels"])
    entity_label_mask = [0] * len(FINDING_NAMES)

    for condition_idx, condition in enumerate(PROMPTMRG_CONDITIONS):
        finding = PROMPTMRG_FINDING_MAP.get(condition)
        if finding is None or condition_idx >= len(promptmrg_labels):
            continue
        finding_idx = FINDING_NAMES.index(finding)
        state = int(promptmrg_labels[condition_idx])
        if state == 0:
            continue
        entity_label_mask[finding_idx] = 1
        entity_labels[finding_idx] = 1 if state == 1 else 0

    merged["entity_labels"] = entity_labels
    merged["plan_finding_prompt"] = list(entity_labels)
    merged["entity_label_mask"] = entity_label_mask
    merged["promptmrg_labels"] = list(promptmrg_labels)
    return merged


def _positive_findings_from_plan(plan: Sequence[Dict[str, object]]) -> set:
    return {
        str(item["finding"])
        for item in plan
        if str(item.get("polarity", "positive")) != "negative"
    }


def _findings_from_text(report: str) -> set:
    text = normalize_report(report)
    return {
        finding
        for finding, terms in FINDING_RULES.items()
        if _contains_any(text, terms) and _finding_polarity(text, terms) != "negative"
    }


def compute_plan_report_consistency(plan: Sequence[Dict[str, object]], generated_report: str) -> float:
    generated = _findings_from_text(generated_report)
    if not generated:
        return 1.0 if not _positive_findings_from_plan(plan) else 0.0
    supported = _positive_findings_from_plan(plan)
    return len(generated & supported) / max(len(generated), 1)


def compute_evidence_support_rate(plan: Sequence[Dict[str, object]], generated_report: str) -> float:
    generated = _findings_from_text(generated_report)
    if not generated:
        return 1.0
    supported = {
        str(item["finding"])
        for item in plan
        if str(item.get("polarity", "positive")) != "negative"
        and float(item.get("evidence_score", 0.0)) > 0.0
    }
    return len(generated & supported) / max(len(generated), 1)


def compute_entity_f1(target: Sequence[int], prediction: Sequence[int]) -> float:
    tp = sum(1 for y, y_hat in zip(target, prediction) if y == 1 and y_hat == 1)
    fp = sum(1 for y, y_hat in zip(target, prediction) if y == 0 and y_hat == 1)
    fn = sum(1 for y, y_hat in zip(target, prediction) if y == 1 and y_hat == 0)
    if tp == 0:
        return 1.0 if fp == 0 and fn == 0 else 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)
