#!/usr/bin/env python3
"""Compute evidence-plan-report consistency metrics from prediction JSON files.

Expected per-example fields:
  - reference / ground_truth / gt / report
  - prediction / generated_report / generated / pred
  - plan / structured_plan / clinical_plan
  - optional evidence / evidence_plan / evidence_findings

The script also understands common wrapper shapes such as:
  {"examples": [...]}, {"rows": [...]}, {"cases": [...]},
  {"results": {"method": {"examples": [...]}}}
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set


try:
    from datasets.clinical_plan_builder import (  # type: ignore
        FINDING_RULES,
        _contains_any,
        _finding_polarity,
        normalize_report,
    )
except Exception:
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


TEXT_KEYS = {
    "reference": ("reference", "ground_truth", "ground_truth_report", "gt", "report", "target"),
    "prediction": ("prediction", "generated_report", "generated", "pred", "hypothesis", "output"),
}
PLAN_KEYS = ("plan", "structured_plan", "clinical_plan", "predicted_plan")
EVIDENCE_KEYS = ("evidence", "evidence_plan", "evidence_findings", "supported_findings")


def _pick(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def findings_from_text(text: Any, include_uncertain: bool = True) -> Set[str]:
    if not isinstance(text, str) or not text.strip():
        return set()
    normalized = normalize_report(text)
    findings = set()
    for finding, terms in FINDING_RULES.items():
        if not _contains_any(normalized, terms):
            continue
        polarity = _finding_polarity(normalized, terms)
        if polarity == "negative":
            continue
        if polarity == "uncertain" and not include_uncertain:
            continue
        findings.add(finding)
    return findings


def findings_from_plan(plan: Any, include_uncertain: bool = True) -> Set[str]:
    if plan is None:
        return set()
    if isinstance(plan, str):
        return findings_from_text(plan, include_uncertain=include_uncertain)
    if isinstance(plan, Mapping):
        plan = [plan]
    if not isinstance(plan, Sequence):
        return set()

    findings = set()
    for item in plan:
        if isinstance(item, str):
            findings |= findings_from_text(item, include_uncertain=include_uncertain)
            continue
        if not isinstance(item, Mapping):
            continue
        finding = item.get("finding") or item.get("label") or item.get("name")
        if finding is None:
            continue
        finding = str(finding).strip().lower().replace(" ", "_")
        polarity = str(item.get("polarity", item.get("status", "positive"))).lower()
        if polarity in {"negative", "neg", "absent", "0", "none"}:
            continue
        if polarity in {"uncertain", "possible"} and not include_uncertain:
            continue
        if finding in FINDING_RULES:
            findings.add(finding)
    return findings


def compute_example(row: Mapping[str, Any], include_uncertain: bool = True) -> Dict[str, Any]:
    reference = _pick(row, TEXT_KEYS["reference"])
    prediction = _pick(row, TEXT_KEYS["prediction"])
    plan = _pick(row, PLAN_KEYS)
    evidence = _pick(row, EVIDENCE_KEYS)

    plan_findings = findings_from_plan(plan, include_uncertain=include_uncertain)
    report_findings = findings_from_text(prediction, include_uncertain=include_uncertain)
    reference_findings = findings_from_text(reference, include_uncertain=include_uncertain)
    evidence_findings = findings_from_plan(evidence, include_uncertain=include_uncertain)
    support_findings = plan_findings | evidence_findings
    if not support_findings:
        support_findings = plan_findings

    plan_coverage = len(plan_findings & report_findings) / len(plan_findings) if plan_findings else 1.0
    report_plan_consistency = (
        len(report_findings & plan_findings) / len(report_findings) if report_findings else 1.0
    )
    unsupported_rate = (
        len(report_findings - support_findings) / len(report_findings) if report_findings else 0.0
    )
    missing_rate = (
        len(reference_findings - report_findings) / len(reference_findings) if reference_findings else 0.0
    )

    return {
        "plan_coverage": plan_coverage,
        "report_plan_consistency": report_plan_consistency,
        "unsupported_rate": unsupported_rate,
        "missing_rate": missing_rate,
        "counts": {
            "plan": len(plan_findings),
            "report": len(report_findings),
            "reference": len(reference_findings),
            "evidence": len(evidence_findings),
            "plan_report_overlap": len(plan_findings & report_findings),
            "unsupported": len(report_findings - support_findings),
            "missing": len(reference_findings - report_findings),
        },
        "sets": {
            "plan": sorted(plan_findings),
            "report": sorted(report_findings),
            "reference": sorted(reference_findings),
            "evidence": sorted(evidence_findings),
        },
    }


def _extract_rows(obj: Any) -> List[Mapping[str, Any]]:
    if isinstance(obj, list):
        return [item for item in obj if isinstance(item, Mapping)]
    if not isinstance(obj, Mapping):
        return []

    for key in ("rows", "examples", "cases", "samples", "predictions"):
        value = obj.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]

    if isinstance(obj.get("results"), Mapping):
        rows: List[Mapping[str, Any]] = []
        for method, metrics in obj["results"].items():
            if not isinstance(metrics, Mapping) or not isinstance(metrics.get("examples"), list):
                continue
            for row in metrics["examples"]:
                if isinstance(row, Mapping):
                    enriched = dict(row)
                    enriched.setdefault("method", method)
                    rows.append(enriched)
        return rows

    if all(key in obj for key in ("reference", "prediction")):
        return [obj]
    return []


def summarize(rows: Sequence[Mapping[str, Any]], include_uncertain: bool = True) -> Dict[str, Any]:
    per_example = [compute_example(row, include_uncertain=include_uncertain) for row in rows]
    metric_keys = ("plan_coverage", "report_plan_consistency", "unsupported_rate", "missing_rate")
    summary = {
        key: float(mean(example[key] for example in per_example)) if per_example else 0.0
        for key in metric_keys
    }

    totals = {
        "plan": sum(item["counts"]["plan"] for item in per_example),
        "report": sum(item["counts"]["report"] for item in per_example),
        "reference": sum(item["counts"]["reference"] for item in per_example),
        "plan_report_overlap": sum(item["counts"]["plan_report_overlap"] for item in per_example),
        "unsupported": sum(item["counts"]["unsupported"] for item in per_example),
        "missing": sum(item["counts"]["missing"] for item in per_example),
    }
    micro = {
        "plan_coverage": totals["plan_report_overlap"] / totals["plan"] if totals["plan"] else 1.0,
        "report_plan_consistency": (
            totals["plan_report_overlap"] / totals["report"] if totals["report"] else 1.0
        ),
        "unsupported_rate": totals["unsupported"] / totals["report"] if totals["report"] else 0.0,
        "missing_rate": totals["missing"] / totals["reference"] if totals["reference"] else 0.0,
    }
    return {
        "num_examples": len(per_example),
        "macro_average": summary,
        "micro_average": micro,
        "totals": totals,
        "per_example": per_example,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Prediction JSON file.")
    parser.add_argument("--output", default=None, help="Optional path to write JSON metrics.")
    parser.add_argument(
        "--exclude-uncertain",
        action="store_true",
        help="Treat uncertain/possible findings as not clinically positive.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Print a compact human-readable summary in addition to JSON output.",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as handle:
        obj = json.load(handle)
    rows = _extract_rows(obj)
    if not rows:
        raise SystemExit(
            "No examples found. Expected a list, or a dict with rows/examples/cases/results."
        )

    result = summarize(rows, include_uncertain=not args.exclude_uncertain)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)

    if args.pretty:
        avg = result["macro_average"]
        print(f"Examples: {result['num_examples']}")
        print(f"Plan Coverage: {avg['plan_coverage']:.4f}")
        print(f"Report-Plan Consistency: {avg['report_plan_consistency']:.4f}")
        print(f"Unsupported Rate: {avg['unsupported_rate']:.4f}")
        print(f"Missing Rate: {avg['missing_rate']:.4f}")
    else:
        print(json.dumps(result["macro_average"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
