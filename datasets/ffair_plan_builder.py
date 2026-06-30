import re
from typing import Dict, Iterable, List, Sequence


FFAIR_FINDING_RULES = {
    "optic_disc_staining": ["optic disc staining", "disc staining", "disc fluorescence", "disc leakage"],
    "vessel_tortuosity": ["tortuous", "tortuosity", "twisted"],
    "delayed_venous_return": ["delayed venous return", "delayed venous reflux", "venous return time is delayed"],
    "capillary_dilation": ["capillary dilation", "capillary expansion", "capillary dilatation", "capillary telangiectasia"],
    "vascular_leakage": ["leakage", "fluorescence leakage", "increased permeability", "permeability"],
    "non_perfusion": ["non-perfusion", "non perfusion", "vascular closure", "capillary closure", "closure area"],
    "retinal_edema": ["retinal edema", "macular edema", "edema"],
    "hemorrhage": ["hemorrhage", "haemorrhage", "bleeding", "pinpoint hemorrhages"],
    "microaneurysm": ["microaneurysm", "microaneurysms"],
    "vitreous_opacity": ["vitreous opacity", "vitreous opacities", "obscures the fluorescence"],
    "arterial_narrowing": ["arteries are slender", "arterial narrowing", "narrow retinal artery", "retinal arteries are slender"],
}

FFAIR_ANATOMY_RULES = {
    "optic_disc": ["optic disc", "disc", "papillary", "pre-papillary"],
    "macula": ["macula", "macular", "fovea"],
    "retinal_artery": ["retinal artery", "retinal arteries", "arterial"],
    "retinal_vein": ["retinal vein", "retinal veins", "venous", "vein"],
    "periphery": ["periphery", "peripheral", "mid-peripheral", "surrounding retina"],
    "temporal_region": ["temporal", "inferior temporal", "superior temporal"],
    "inferior_region": ["inferior", "lower half", "lower"],
    "posterior_pole": ["posterior pole", "posterior"],
}

NEGATIVE_PREFIXES = (
    "no",
    "without",
    "negative for",
    "absence of",
    "free of",
)

UNCERTAIN_PREFIXES = (
    "possible",
    "possibly",
    "suspected",
    "suspicious",
    "may represent",
    "cannot exclude",
)

FFAIR_FINDING_NAMES = list(FFAIR_FINDING_RULES.keys())
FFAIR_ANATOMY_NAMES = list(FFAIR_ANATOMY_RULES.keys())
POLARITY_TO_ID = {"negative": 0, "positive": 1, "uncertain": 2}


def normalize_ffair_report(text: str) -> str:
    text = str(text or "").lower().replace("\n", " ")
    text = text.replace("ffa", " fluorescein fundus angiography ")
    text = re.sub(r"[^a-z0-9\s.\-]", " ", text)
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


def extract_ffair_structured_plan(report: str) -> List[Dict[str, object]]:
    text = normalize_ffair_report(report)
    matched_anatomies = [
        anatomy for anatomy, terms in FFAIR_ANATOMY_RULES.items() if _contains_any(text, terms)
    ]
    if not matched_anatomies:
        matched_anatomies = ["posterior_pole"]

    plan = []
    for finding, terms in FFAIR_FINDING_RULES.items():
        if not _contains_any(text, terms):
            continue
        polarity = _finding_polarity(text, terms)
        anatomies = matched_anatomies
        if finding == "optic_disc_staining" and "optic_disc" not in anatomies:
            anatomies = ["optic_disc"]
        elif finding == "delayed_venous_return" and "retinal_vein" not in anatomies:
            anatomies = ["retinal_vein"]
        elif finding == "arterial_narrowing" and "retinal_artery" not in anatomies:
            anatomies = ["retinal_artery"]
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


def build_ffair_clinical_sample(report: str) -> Dict[str, object]:
    plan = extract_ffair_structured_plan(report)
    entity_labels = [0] * len(FFAIR_FINDING_NAMES)
    anatomy_finding_labels = [[0] * len(FFAIR_FINDING_NAMES) for _ in FFAIR_ANATOMY_NAMES]
    polarity_labels = [POLARITY_TO_ID["negative"]] * len(FFAIR_FINDING_NAMES)

    for item in plan:
        finding_id = FFAIR_FINDING_NAMES.index(str(item["finding"]))
        anatomy_id = FFAIR_ANATOMY_NAMES.index(str(item["anatomy"]))
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
        "entity_label_mask": [1] * len(FFAIR_FINDING_NAMES),
        "anatomy_finding_labels": anatomy_finding_labels,
        "polarity_labels": polarity_labels,
    }


def _positive_findings_from_plan(plan: Sequence[Dict[str, object]]) -> set:
    return {
        str(item["finding"])
        for item in plan
        if str(item.get("polarity", "positive")) != "negative"
    }


def _findings_from_text(report: str) -> set:
    text = normalize_ffair_report(report)
    return {
        finding
        for finding, terms in FFAIR_FINDING_RULES.items()
        if _contains_any(text, terms) and _finding_polarity(text, terms) != "negative"
    }


def compute_ffair_plan_report_consistency(plan: Sequence[Dict[str, object]], generated_report: str) -> float:
    generated = _findings_from_text(generated_report)
    if not generated:
        return 1.0 if not _positive_findings_from_plan(plan) else 0.0
    supported = _positive_findings_from_plan(plan)
    return len(generated & supported) / max(len(generated), 1)


def compute_ffair_evidence_support_rate(plan: Sequence[Dict[str, object]], generated_report: str) -> float:
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
