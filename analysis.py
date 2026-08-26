#!/usr/bin/env python3
"""Reproducible evaluation pipeline for complex-PCI extraction with LLMs.

This is the public-facing analysis code accompanying the manuscript on automated
identification of complex percutaneous coronary intervention (PCI) from cardiac
catheterization reports using large language models.

The patient-level source data and adjudicated labels are not distributed with the
repository because of IRB/privacy constraints. This script therefore starts from:

1. A *final adjudicated gold-standard CSV* available in the local/private analysis
   environment.
2. Two JSON output files per model:
      - PCI identification output ("prompt 1" in the development notebook)
      - complex-PCI variable extraction output ("prompt 3")

The script performs only the minimal preparation needed to make the evaluation
DataFrames, then reproduces the manuscript-facing analyses:

- PCI identification: overall and by clinical site
- Exact-match extraction accuracy for six complex-PCI variables
- Complex-PCI classification: overall and by clinical site
- Binary performance for each of the six complex-PCI criteria
- Wilson 95% confidence intervals for proportions
- Site-level heterogeneity for accuracy, sensitivity, and specificity using
  Pearson chi-square tests plus an I2-style statistic
- Main/supplementary figures and source-data tables

Expected gold-standard columns
------------------------------
output_number
Site
PCI_performed
vessels_treated
lesions_treated
stents_implanted
total_stent_length_mm
bifurcation_with_2_stents
chronic_total_occlusion

Additional columns in the private gold-standard file are ignored by the analysis.
The Narrative text itself is not required for this evaluation script.

JSON format
-----------
The parser intentionally accepts the common formats used during the study,
including:

- a list of JSON records;
- a mapping from row/output number -> record;
- a mapping from row/output number -> JSON-formatted string;
- a single wrapper such as {"prompt1_output": {"0": "{...}"}};
- records with identifiers named output_number, _output_number, or row_index;
- records containing a JSON response inside response/output/generated_text/text.

Example
-------
python analysis.py \
    --gold-standard /private/path/gold_standard.csv \
    --model "Llama 3 70B" /private/path/llama_prompt1.json /private/path/llama_prompt3.json \
    --model "Meditron-7B" /private/path/meditron_prompt1.json /private/path/meditron_prompt3.json \
    --model "BioMistral-7B" /private/path/biomistral_prompt1.json /private/path/biomistral_prompt3.json \
    --outdir results

The --model argument can be repeated for any number of models and preserves the
order supplied on the command line.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.stats import chi2
from sklearn.metrics import confusion_matrix
from statsmodels.stats.proportion import proportion_confint


# =============================================================================
# STUDY DEFINITIONS
# =============================================================================

ID_COL = "output_number"
SITE_COL = "Site"
PCI_TRUE_COL = "PCI_performed"
PCI_PRED_COL = "PCI_predicted"

NUMERIC_COMPONENTS = [
    "vessels_treated",
    "lesions_treated",
    "stents_implanted",
    "total_stent_length_mm",
]

BINARY_COMPONENTS = [
    "bifurcation_with_2_stents",
    "chronic_total_occlusion",
]

COMPONENT_ORDER = NUMERIC_COMPONENTS + BINARY_COMPONENTS

# Giustino-style complex-PCI criteria used in the manuscript.
# Total stent length is >60 mm in the current analysis.
COMPLEX_COMPONENTS: Dict[str, Dict[str, Any]] = {
    "vessels_treated": {
        "gold": "vessels_treated",
        "pred": "vessels_treated_predicted",
        "type": "numeric",
        "threshold": 3,
        "operator": ">=",
        "label": "Vessels treated ≥3",
    },
    "lesions_treated": {
        "gold": "lesions_treated",
        "pred": "lesions_treated_predicted",
        "type": "numeric",
        "threshold": 3,
        "operator": ">=",
        "label": "Lesions treated ≥3",
    },
    "stents_implanted": {
        "gold": "stents_implanted",
        "pred": "stents_implanted_predicted",
        "type": "numeric",
        "threshold": 3,
        "operator": ">=",
        "label": "Stents implanted ≥3",
    },
    "total_stent_length_mm": {
        "gold": "total_stent_length_mm",
        "pred": "total_stent_length_mm_predicted",
        "type": "numeric",
        "threshold": 60,
        "operator": ">",
        "label": "Total stent length >60 mm",
    },
    "bifurcation_with_2_stents": {
        "gold": "bifurcation_with_2_stents",
        "pred": "bifurcation_with_2_stents_predicted",
        "type": "binary",
        "label": "Bifurcation with 2 stents",
    },
    "chronic_total_occlusion": {
        "gold": "chronic_total_occlusion",
        "pred": "chronic_total_occlusion_predicted",
        "type": "binary",
        "label": "Chronic total occlusion",
    },
}

EXTRACTION_LABELS = {
    "vessels_treated": "Vessels treated",
    "lesions_treated": "Lesions treated",
    "stents_implanted": "Stents implanted",
    "total_stent_length_mm": "Total stent length",
    "bifurcation_with_2_stents": "Bifurcation with 2 stents",
    "chronic_total_occlusion": "Chronic total occlusion",
}

PERFORMANCE_METRICS = [
    "Sensitivity",
    "Specificity",
    "PPV",
    "NPV",
    "Accuracy",
    "F1",
]

HETEROGENEITY_METRICS = ["Accuracy", "Sensitivity", "Specificity"]
CONFUSION_COUNT_ORDER = ["TP", "FP", "TN", "FN"]

# Presentation-only site abbreviations used in the manuscript figures.
# Unknown site names are retained unchanged.
SITE_LABEL_MAP = {
    "Yale New Haven Hospital": "YNHH",
    "Lawrence and Memorial Hospital": "LM",
    "Lawrence + Memorial Hospital": "LM",
    "Bridgeport Hospital": "BH",
}

MISSING_TEXT = {
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "missing",
    "missing site",
    "unknown",
    "not reported",
    "not available",
}

ID_CANDIDATES = (
    "output_number",
    "_output_number",
    "row_index",
    "record_id",
    "id",
)

RESPONSE_TEXT_KEYS = (
    "response",
    "output",
    "generated_text",
    "completion",
    "text",
)

KNOWN_OUTPUT_FIELDS = {
    "PCI_performed",
    "PCI_predicted",
    "vessels_treated",
    "lesions_treated",
    "stents_implanted",
    "total_stent_length_mm",
    "bifurcation_with_2_stents",
    "chronic_total_occlusion",
    "complex_pci",
    *ID_CANDIDATES,
}


# =============================================================================
# MINIMAL JSON PARSING
# =============================================================================

def strip_code_fences(text: str) -> str:
    """Remove Markdown code fences surrounding model JSON."""
    text = text.replace("\r", "")
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    return text.replace("```", "").strip()


def extract_first_brace_block(text: str) -> Optional[str]:
    """Return the first balanced {...} block from a string, if present."""
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def try_parse_json_record(value: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse one model response into a dictionary without failing on minor formatting."""
    if isinstance(value, Mapping):
        record = dict(value)

        # Some generation pipelines store the actual JSON in a response/text field.
        if not any(field in record for field in KNOWN_OUTPUT_FIELDS):
            for key in RESPONSE_TEXT_KEYS:
                nested = record.get(key)
                if isinstance(nested, str):
                    parsed, _ = try_parse_json_record(nested)
                    if parsed is not None:
                        for id_key in ID_CANDIDATES:
                            if id_key in record and id_key not in parsed:
                                parsed[id_key] = record[id_key]
                        return parsed, None

        return record, None

    if not isinstance(value, str) or not value.strip():
        return None, None

    text = strip_code_fences(value)

    try:
        parsed = json.loads(text)
        return (dict(parsed), None) if isinstance(parsed, Mapping) else (None, text)
    except json.JSONDecodeError:
        pass

    block = extract_first_brace_block(text)
    if block is None:
        return None, text

    attempts = [
        block,
        re.sub(r",(\s*[}\]])", r"\1", block)
        .replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .replace("‘", "'"),
    ]

    for candidate in attempts:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, Mapping):
                return dict(parsed), None
        except json.JSONDecodeError:
            continue

    return None, text


def _looks_like_single_record(obj: Mapping[str, Any]) -> bool:
    return any(key in obj for key in KNOWN_OUTPUT_FIELDS)


def _append_record(
    records: List[Dict[str, Any]],
    value: Any,
    fallback_id: Optional[Any] = None,
) -> None:
    record, raw = try_parse_json_record(value)
    if record is None:
        if raw is None:
            return
        record = {"_raw": raw}

    if fallback_id is not None and not any(key in record for key in ID_CANDIDATES):
        record[ID_COL] = fallback_id

    records.append(record)


def parse_json_file(path: Path) -> pd.DataFrame:
    """Parse a model-output JSON file into one row per report/output."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Unwrap a single named container such as {"prompt1_output": {...}}.
    while (
        isinstance(data, Mapping)
        and len(data) == 1
        and not _looks_like_single_record(data)
    ):
        only_value = next(iter(data.values()))
        if isinstance(only_value, (Mapping, list)):
            data = only_value
        else:
            break

    records: List[Dict[str, Any]] = []

    if isinstance(data, list):
        for value in data:
            _append_record(records, value)

    elif isinstance(data, Mapping):
        if _looks_like_single_record(data):
            _append_record(records, data)
        else:
            # Prefer an obvious embedded list of records, if present.
            list_values = [
                value
                for value in data.values()
                if isinstance(value, list) and value
            ]
            list_of_records = [
                value
                for value in list_values
                if all(isinstance(item, (Mapping, str)) for item in value)
            ]

            if len(list_of_records) == 1:
                for value in list_of_records[0]:
                    _append_record(records, value)
            else:
                # Otherwise interpret the outer mapping as output_id -> response.
                for output_id, value in data.items():
                    _append_record(records, value, fallback_id=output_id)
    else:
        raise ValueError(f"Unsupported JSON structure in {path}")

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No records could be parsed from {path}")

    return standardize_identifier(df, source_name=str(path))


def normalize_identifier_value(value: Any) -> Optional[str]:
    """Normalize integer-like identifiers so 12, 12.0, and '12' merge consistently."""
    if pd.isna(value):
        return None

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))

    text = str(value).strip()
    if re.fullmatch(r"[-+]?\d+\.0+", text):
        return text.split(".", 1)[0]

    return text if text else None


def standardize_identifier(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Standardize a supported identifier column to output_number."""
    d = df.copy()

    id_col = next((candidate for candidate in ID_CANDIDATES if candidate in d.columns), None)
    if id_col is None:
        raise KeyError(
            f"{source_name}: could not find a report identifier. "
            f"Expected one of {ID_CANDIDATES}."
        )

    if id_col != ID_COL:
        d = d.rename(columns={id_col: ID_COL})

    d[ID_COL] = d[ID_COL].map(normalize_identifier_value)
    d = d.dropna(subset=[ID_COL]).copy()

    duplicated = d[ID_COL].duplicated(keep=False)
    if duplicated.any():
        examples = d.loc[duplicated, ID_COL].astype(str).unique()[:5]
        raise ValueError(
            f"{source_name}: duplicate {ID_COL} values found after parsing; "
            f"examples: {', '.join(examples)}. Resolve duplicates upstream."
        )

    return d


def prepare_pci_prediction_json(path: Path) -> pd.DataFrame:
    """Parse PCI-identification JSON and return output_number + PCI_predicted."""
    d = parse_json_file(path)

    if PCI_PRED_COL in d.columns:
        source_col = PCI_PRED_COL
    elif PCI_TRUE_COL in d.columns:
        source_col = PCI_TRUE_COL
    else:
        raise KeyError(
            f"{path}: PCI output must contain '{PCI_TRUE_COL}' or '{PCI_PRED_COL}'."
        )

    out = d[[ID_COL, source_col]].copy()
    out = out.rename(columns={source_col: PCI_PRED_COL})
    return out


def prepare_extraction_prediction_json(path: Path) -> pd.DataFrame:
    """Parse complex-PCI extraction JSON and standardize predicted field names."""
    d = parse_json_file(path)
    out = pd.DataFrame({ID_COL: d[ID_COL]})

    missing_fields: List[str] = []

    for component in COMPONENT_ORDER:
        predicted_name = f"{component}_predicted"
        if predicted_name in d.columns:
            out[predicted_name] = d[predicted_name]
        elif component in d.columns:
            out[predicted_name] = d[component]
        else:
            out[predicted_name] = np.nan
            missing_fields.append(component)

    if missing_fields:
        print(
            f"Warning: {path.name} does not contain: {', '.join(missing_fields)}. "
            "Those predictions will be treated as missing."
        )

    return out


def prepare_model_dataframe(
    gold: pd.DataFrame,
    pci_json: Path,
    extraction_json: Path,
    model_name: str,
) -> pd.DataFrame:
    """Merge one model's two JSON outputs with the final adjudicated gold standard."""
    pci_pred = prepare_pci_prediction_json(pci_json)
    extraction_pred = prepare_extraction_prediction_json(extraction_json)

    predictions = pci_pred.merge(extraction_pred, on=ID_COL, how="outer")
    merged = gold.merge(predictions, on=ID_COL, how="left")
    merged["Model"] = model_name

    if SITE_COL not in merged.columns:
        merged[SITE_COL] = np.nan

    merged[SITE_COL] = merged[SITE_COL].map(clean_site_label)
    return add_complex_pci_columns(merged)


# =============================================================================
# BASIC CLEANING AND METRICS
# =============================================================================

def to01_yesno(value: Any) -> float:
    """Map common yes/no encodings to 1/0; unknown/unparseable values become NaN."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()
    if text in {"yes", "y", "1", "true", "t"}:
        return 1.0
    if text in {"no", "n", "0", "false", "f"}:
        return 0.0
    return np.nan


def to_numeric_clean(value: Any) -> float:
    """Convert numeric-looking model outputs to float; unknown values become NaN."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip().lower()
    if text in MISSING_TEXT:
        return np.nan

    text = text.replace("mm", "").replace(",", "").strip()

    try:
        return float(text)
    except ValueError:
        numbers = re.findall(r"[-+]?\d*\.?\d+", text)
        return float(numbers[0]) if numbers else np.nan


def is_missing_site(value: Any) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in MISSING_TEXT


def clean_site_label(value: Any) -> Any:
    """Apply manuscript abbreviations while preserving any unrecognized site label."""
    if is_missing_site(value):
        return np.nan
    text = str(value).strip()
    return SITE_LABEL_MAP.get(text, text)


def drop_missing_sites(df: pd.DataFrame) -> pd.DataFrame:
    if SITE_COL not in df.columns:
        return df.iloc[0:0].copy()
    d = df.dropna(subset=[SITE_COL]).copy()
    return d[~d[SITE_COL].map(is_missing_site)].copy()


def require_columns(df: pd.DataFrame, columns: Iterable[str], source_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"{source_name}: missing required columns: {', '.join(missing)}")


def load_gold_standard(path: Path) -> pd.DataFrame:
    """Load the already-finalized/adjudicated gold-standard CSV."""
    gold = pd.read_csv(path)
    require_columns(
        gold,
        [ID_COL, SITE_COL, PCI_TRUE_COL, *COMPONENT_ORDER],
        source_name=str(path),
    )
    gold = standardize_identifier(gold, source_name=str(path))
    gold[SITE_COL] = gold[SITE_COL].map(clean_site_label)
    return gold


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    low, high = proportion_confint(int(k), int(n), alpha=alpha, method="wilson")
    return float(low), float(high)


def proportion_with_ci(k: int, n: int) -> Tuple[float, float, float]:
    if n <= 0:
        return np.nan, np.nan, np.nan
    value = k / n
    low, high = wilson_ci(k, n)
    return float(value), low, high


def prep_binary_pairs(df: pd.DataFrame, true_col: str, pred_col: str) -> pd.DataFrame:
    d = df.copy()
    d["y_true"] = d[true_col].map(to01_yesno)
    d["y_pred"] = d[pred_col].map(to01_yesno)
    d = d.dropna(subset=["y_true", "y_pred"]).copy()
    d["y_true"] = d["y_true"].astype(int)
    d["y_pred"] = d["y_pred"].astype(int)
    return d


def binary_metrics(pairs: pd.DataFrame) -> Dict[str, Any]:
    """Sensitivity, specificity, PPV, NPV, accuracy, F1, and confusion counts."""
    if pairs.empty:
        return {
            "n_eval": 0,
            "TP": np.nan,
            "FP": np.nan,
            "TN": np.nan,
            "FN": np.nan,
            "Sensitivity": np.nan,
            "Sensitivity_low": np.nan,
            "Sensitivity_high": np.nan,
            "Specificity": np.nan,
            "Specificity_low": np.nan,
            "Specificity_high": np.nan,
            "PPV": np.nan,
            "PPV_low": np.nan,
            "PPV_high": np.nan,
            "NPV": np.nan,
            "NPV_low": np.nan,
            "NPV_high": np.nan,
            "Accuracy": np.nan,
            "Accuracy_low": np.nan,
            "Accuracy_high": np.nan,
            "F1": np.nan,
        }

    tn, fp, fn, tp = confusion_matrix(
        pairs["y_true"], pairs["y_pred"], labels=[0, 1]
    ).ravel()

    sensitivity, sensitivity_low, sensitivity_high = proportion_with_ci(tp, tp + fn)
    specificity, specificity_low, specificity_high = proportion_with_ci(tn, tn + fp)
    ppv, ppv_low, ppv_high = proportion_with_ci(tp, tp + fp)
    npv, npv_low, npv_high = proportion_with_ci(tn, tn + fn)
    accuracy, accuracy_low, accuracy_high = proportion_with_ci(
        tp + tn, tp + tn + fp + fn
    )
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan

    return {
        "n_eval": int(len(pairs)),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "Sensitivity": sensitivity,
        "Sensitivity_low": sensitivity_low,
        "Sensitivity_high": sensitivity_high,
        "Specificity": specificity,
        "Specificity_low": specificity_low,
        "Specificity_high": specificity_high,
        "PPV": ppv,
        "PPV_low": ppv_low,
        "PPV_high": ppv_high,
        "NPV": npv,
        "NPV_low": npv_low,
        "NPV_high": npv_high,
        "Accuracy": accuracy,
        "Accuracy_low": accuracy_low,
        "Accuracy_high": accuracy_high,
        "F1": float(f1) if pd.notna(f1) else np.nan,
    }


def filter_true_pci_cases(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["_gold_pci"] = d[PCI_TRUE_COL].map(to01_yesno)
    return d[d["_gold_pci"] == 1].copy()


def exact_match_accuracy(
    df: pd.DataFrame,
    gold_col: str,
    pred_col: str,
    value_type: str,
) -> Dict[str, Any]:
    """Exact-match extraction accuracy using directly normalized gold/predicted values."""
    if gold_col not in df.columns or pred_col not in df.columns:
        return {
            "n_eval": 0,
            "n_correct": np.nan,
            "n_incorrect": np.nan,
            "Accuracy": np.nan,
            "Accuracy_low": np.nan,
            "Accuracy_high": np.nan,
        }

    if value_type == "numeric":
        gold = df[gold_col].map(to_numeric_clean)
        pred = df[pred_col].map(to_numeric_clean)
    elif value_type == "binary":
        gold = df[gold_col].map(to01_yesno)
        pred = df[pred_col].map(to01_yesno)
    else:
        raise ValueError(f"Unsupported value_type: {value_type}")

    evaluable = gold.notna() & pred.notna()
    n = int(evaluable.sum())
    if n == 0:
        return {
            "n_eval": 0,
            "n_correct": np.nan,
            "n_incorrect": np.nan,
            "Accuracy": np.nan,
            "Accuracy_low": np.nan,
            "Accuracy_high": np.nan,
        }

    k = int((gold[evaluable] == pred[evaluable]).sum())
    accuracy, low, high = proportion_with_ci(k, n)
    return {
        "n_eval": n,
        "n_correct": k,
        "n_incorrect": n - k,
        "Accuracy": accuracy,
        "Accuracy_low": low,
        "Accuracy_high": high,
    }


# =============================================================================
# COMPLEX-PCI DERIVATION
# =============================================================================

def compare_threshold(value: Any, threshold: float, operator: str) -> float:
    numeric = to_numeric_clean(value)
    if pd.isna(numeric):
        return np.nan
    if operator == ">=":
        return float(numeric >= threshold)
    if operator == ">":
        return float(numeric > threshold)
    if operator == "<=":
        return float(numeric <= threshold)
    if operator == "<":
        return float(numeric < threshold)
    raise ValueError(f"Unsupported threshold operator: {operator}")


def add_complex_component_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    d = df.copy()

    for component, spec in COMPLEX_COMPONENTS.items():
        source_col = spec[prefix]
        out_col = f"{prefix}_{component}_criterion"

        if source_col not in d.columns:
            d[out_col] = np.nan
        elif spec["type"] == "numeric":
            d[out_col] = d[source_col].map(
                lambda value, s=spec: compare_threshold(
                    value, s["threshold"], s["operator"]
                )
            )
        else:
            d[out_col] = d[source_col].map(to01_yesno)

    return d


def derive_complex_pci_status(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """1 if any criterion positive; 0 only if all six are known and negative."""
    d = df.copy()
    criterion_cols = [f"{prefix}_{component}_criterion" for component in COMPONENT_ORDER]

    def classify(row: pd.Series) -> float:
        values = row[criterion_cols]
        if (values == 1).any():
            return 1.0
        if values.isna().any():
            return np.nan
        return 0.0

    d[f"{prefix}_complex_pci"] = d.apply(classify, axis=1)
    return d


def add_complex_pci_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = add_complex_component_columns(df, "gold")
    d = add_complex_component_columns(d, "pred")
    d = derive_complex_pci_status(d, "gold")
    d = derive_complex_pci_status(d, "pred")
    return d


def complex_pci_pairs(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["gold_complex_pci", "pred_complex_pci"]).copy()
    d["y_true"] = d["gold_complex_pci"].astype(int)
    d["y_pred"] = d["pred_complex_pci"].astype(int)
    return d


def component_pairs(df: pd.DataFrame, component: str) -> pd.DataFrame:
    gold_col = f"gold_{component}_criterion"
    pred_col = f"pred_{component}_criterion"
    d = df.dropna(subset=[gold_col, pred_col]).copy()
    d["y_true"] = d[gold_col].astype(int)
    d["y_pred"] = d[pred_col].astype(int)
    return d


# =============================================================================
# SITE-LEVEL HETEROGENEITY
# =============================================================================

def site_kn_for_metric(pairs: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return site-specific success count k and denominator n for one metric."""
    d = drop_missing_sites(pairs)
    rows: List[Dict[str, Any]] = []

    for site, sub in d.groupby(SITE_COL, dropna=True):
        if metric == "Accuracy":
            n = len(sub)
            k = int((sub["y_true"] == sub["y_pred"]).sum())
        elif metric == "Sensitivity":
            denominator = sub[sub["y_true"] == 1]
            n = len(denominator)
            k = int((denominator["y_pred"] == 1).sum())
        elif metric == "Specificity":
            denominator = sub[sub["y_true"] == 0]
            n = len(denominator)
            k = int((denominator["y_pred"] == 0).sum())
        else:
            raise ValueError(f"Unsupported heterogeneity metric: {metric}")

        rows.append(
            {
                SITE_COL: site,
                "Metric": metric,
                "k": k,
                "n": n,
                "Proportion": k / n if n > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def heterogeneity_chisq(kn_df: pd.DataFrame) -> Dict[str, Any]:
    """Pearson chi-square test for equality of metric-specific proportions across sites."""
    d = kn_df.dropna(subset=["k", "n"]).copy()
    d = d[d["n"] > 0].copy()

    if len(d) < 2:
        return {
            "n_sites": int(len(d)),
            "total_n": int(d["n"].sum()) if len(d) else 0,
            "Q": np.nan,
            "df": np.nan,
            "p_heterogeneity": np.nan,
            "I2_percent": np.nan,
        }

    total_k = float(d["k"].sum())
    total_n = float(d["n"].sum())
    pooled = total_k / total_n

    # If every evaluable observation is a success or every one is a failure,
    # the Pearson statistic is not estimable (reported as NE in the manuscript).
    if pooled <= 0 or pooled >= 1:
        return {
            "n_sites": int(len(d)),
            "total_n": int(total_n),
            "Q": np.nan,
            "df": int(len(d) - 1),
            "p_heterogeneity": np.nan,
            "I2_percent": np.nan,
        }

    q_stat = 0.0
    for _, row in d.iterrows():
        n_i = float(row["n"])
        k_i = float(row["k"])
        fail_i = n_i - k_i

        expected_k = n_i * pooled
        expected_fail = n_i * (1 - pooled)

        q_stat += ((k_i - expected_k) ** 2) / expected_k
        q_stat += ((fail_i - expected_fail) ** 2) / expected_fail

    df_value = len(d) - 1
    p_value = float(chi2.sf(q_stat, df_value))
    i2 = max(0.0, (q_stat - df_value) / q_stat) * 100 if q_stat > 0 else 0.0

    return {
        "n_sites": int(len(d)),
        "total_n": int(total_n),
        "Q": float(q_stat),
        "df": int(df_value),
        "p_heterogeneity": p_value,
        "I2_percent": float(i2),
    }


def run_site_heterogeneity(
    pairs: pd.DataFrame,
    model_name: str,
    analysis_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, Any]] = []
    site_rows: List[pd.DataFrame] = []

    for metric in HETEROGENEITY_METRICS:
        kn = site_kn_for_metric(pairs, metric)
        if kn.empty:
            kn = pd.DataFrame(columns=[SITE_COL, "Metric", "k", "n", "Proportion"])

        kn.insert(0, "Analysis", analysis_name)
        kn.insert(1, "Model", model_name)
        site_rows.append(kn)

        result = heterogeneity_chisq(kn)
        result.update({"Analysis": analysis_name, "Model": model_name, "Metric": metric})
        summary_rows.append(result)

    return pd.DataFrame(summary_rows), pd.concat(site_rows, ignore_index=True)


# =============================================================================
# BUILD MANUSCRIPT RESULTS
# =============================================================================

def evaluate_models(
    model_dfs: Mapping[str, pd.DataFrame],
    model_order: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    pci_overall_rows: List[Dict[str, Any]] = []
    pci_site_rows: List[Dict[str, Any]] = []
    extraction_rows: List[Dict[str, Any]] = []
    complex_overall_rows: List[Dict[str, Any]] = []
    complex_site_rows: List[Dict[str, Any]] = []
    component_rows: List[Dict[str, Any]] = []
    heterogeneity_frames: List[pd.DataFrame] = []
    heterogeneity_site_frames: List[pd.DataFrame] = []
    cohort_rows: List[Dict[str, Any]] = []

    for model_name in model_order:
        df = model_dfs[model_name]

        # ---------------------------------------------------------------------
        # Task 1: PCI identification
        # ---------------------------------------------------------------------
        pci_pairs = prep_binary_pairs(df, PCI_TRUE_COL, PCI_PRED_COL)
        pci_metrics = binary_metrics(pci_pairs)
        pci_metrics.update({"Model": model_name, "Analysis": "PCI identification"})
        pci_overall_rows.append(pci_metrics)

        site_pci_pairs = drop_missing_sites(pci_pairs)
        for site, sub in site_pci_pairs.groupby(SITE_COL, dropna=True):
            row = binary_metrics(sub)
            row.update(
                {
                    "Model": model_name,
                    SITE_COL: site,
                    "Analysis": "PCI identification",
                }
            )
            pci_site_rows.append(row)

        het, het_site = run_site_heterogeneity(
            pci_pairs, model_name, "PCI identification"
        )
        heterogeneity_frames.append(het)
        heterogeneity_site_frames.append(het_site)

        # ---------------------------------------------------------------------
        # Task 2: extraction and complex PCI are evaluated among gold PCI cases
        # ---------------------------------------------------------------------
        true_pci_df = filter_true_pci_cases(df)

        for component in COMPONENT_ORDER:
            spec = COMPLEX_COMPONENTS[component]
            row = exact_match_accuracy(
                true_pci_df,
                gold_col=spec["gold"],
                pred_col=spec["pred"],
                value_type=spec["type"],
            )
            row.update(
                {
                    "Model": model_name,
                    "Variable": EXTRACTION_LABELS[component],
                    "Component": component,
                }
            )
            extraction_rows.append(row)

        complex_pairs = complex_pci_pairs(true_pci_df)
        complex_metrics = binary_metrics(complex_pairs)
        complex_metrics.update(
            {"Model": model_name, "Analysis": "Complex PCI classification"}
        )
        complex_overall_rows.append(complex_metrics)

        site_complex_pairs = drop_missing_sites(complex_pairs)
        for site, sub in site_complex_pairs.groupby(SITE_COL, dropna=True):
            row = binary_metrics(sub)
            row.update(
                {
                    "Model": model_name,
                    SITE_COL: site,
                    "Analysis": "Complex PCI classification",
                }
            )
            complex_site_rows.append(row)

        het, het_site = run_site_heterogeneity(
            complex_pairs, model_name, "Complex PCI classification"
        )
        heterogeneity_frames.append(het)
        heterogeneity_site_frames.append(het_site)

        # Criterion-level binary classification; overall only.
        for component in COMPONENT_ORDER:
            pairs = component_pairs(true_pci_df, component)
            row = binary_metrics(pairs)
            row.update(
                {
                    "Model": model_name,
                    "Component": component,
                    "Analysis": f"Criterion: {COMPLEX_COMPONENTS[component]['label']}",
                    "Figure label": COMPLEX_COMPONENTS[component]["label"],
                }
            )
            component_rows.append(row)

        cohort_rows.append(
            {
                "Model": model_name,
                "Gold-standard reports": int(len(df)),
                "Reports with clinical site": int(df[SITE_COL].notna().sum()),
                "Gold-standard PCI reports": int(len(true_pci_df)),
                "PCI identification evaluable reports": int(len(pci_pairs)),
                "PCI identification evaluable reports with site": int(len(site_pci_pairs)),
                "Complex PCI evaluable reports": int(len(complex_pairs)),
                "Complex PCI evaluable reports with site": int(len(site_complex_pairs)),
            }
        )

    results = {
        "cohort": pd.DataFrame(cohort_rows),
        "pci_overall": pd.DataFrame(pci_overall_rows),
        "pci_by_site": pd.DataFrame(pci_site_rows),
        "extraction_accuracy": pd.DataFrame(extraction_rows),
        "complex_overall": pd.DataFrame(complex_overall_rows),
        "complex_by_site": pd.DataFrame(complex_site_rows),
        "component_metrics": pd.DataFrame(component_rows),
        "heterogeneity": pd.concat(heterogeneity_frames, ignore_index=True),
        "heterogeneity_site_kn": pd.concat(heterogeneity_site_frames, ignore_index=True),
    }

    # Stable manuscript ordering.
    for table in results.values():
        if "Model" in table.columns:
            table["Model"] = pd.Categorical(
                table["Model"], categories=list(model_order), ordered=True
            )

    if not results["extraction_accuracy"].empty:
        results["extraction_accuracy"]["Variable"] = pd.Categorical(
            results["extraction_accuracy"]["Variable"],
            categories=[EXTRACTION_LABELS[c] for c in COMPONENT_ORDER],
            ordered=True,
        )

    if not results["component_metrics"].empty:
        results["component_metrics"]["Component"] = pd.Categorical(
            results["component_metrics"]["Component"],
            categories=COMPONENT_ORDER,
            ordered=True,
        )

    if not results["heterogeneity"].empty:
        results["heterogeneity"]["Metric"] = pd.Categorical(
            results["heterogeneity"]["Metric"],
            categories=HETEROGENEITY_METRICS,
            ordered=True,
        )

    if not results["heterogeneity_site_kn"].empty:
        results["heterogeneity_site_kn"]["Metric"] = pd.Categorical(
            results["heterogeneity_site_kn"]["Metric"],
            categories=HETEROGENEITY_METRICS,
            ordered=True,
        )

    return results


# =============================================================================
# TABLE OUTPUTS
# =============================================================================

def format_percent(value: Any) -> str:
    return "" if pd.isna(value) else f"{100 * float(value):.1f}"


def format_ci(low: Any, high: Any) -> str:
    if pd.isna(low) or pd.isna(high):
        return ""
    return f"{100 * float(low):.1f}-{100 * float(high):.1f}"


def make_supplementary_metrics_table(
    pci_overall: pd.DataFrame,
    complex_overall: pd.DataFrame,
    component_metrics: pd.DataFrame,
    model_order: Sequence[str],
) -> pd.DataFrame:
    combined = pd.concat(
        [pci_overall, complex_overall, component_metrics], ignore_index=True, sort=False
    )

    analysis_order = [
        "PCI identification",
        "Complex PCI classification",
        *[f"Criterion: {COMPLEX_COMPONENTS[c]['label']}" for c in COMPONENT_ORDER],
    ]

    combined["Analysis"] = pd.Categorical(
        combined["Analysis"], categories=analysis_order, ordered=True
    )
    combined["Model"] = pd.Categorical(
        combined["Model"], categories=list(model_order), ordered=True
    )
    combined = combined.sort_values(["Analysis", "Model"]).copy()

    out = pd.DataFrame(
        {
            "Analysis": combined["Analysis"].astype(str),
            "Model": combined["Model"].astype(str),
            "Evaluable reports": combined["n_eval"],
            "TP": combined["TP"],
            "FP": combined["FP"],
            "TN": combined["TN"],
            "FN": combined["FN"],
            "Sensitivity, %": combined["Sensitivity"].map(format_percent),
            "Sensitivity 95% CI, %": [
                format_ci(low, high)
                for low, high in zip(
                    combined["Sensitivity_low"], combined["Sensitivity_high"]
                )
            ],
            "Specificity, %": combined["Specificity"].map(format_percent),
            "Specificity 95% CI, %": [
                format_ci(low, high)
                for low, high in zip(
                    combined["Specificity_low"], combined["Specificity_high"]
                )
            ],
            "PPV, %": combined["PPV"].map(format_percent),
            "PPV 95% CI, %": [
                format_ci(low, high)
                for low, high in zip(combined["PPV_low"], combined["PPV_high"])
            ],
            "NPV, %": combined["NPV"].map(format_percent),
            "NPV 95% CI, %": [
                format_ci(low, high)
                for low, high in zip(combined["NPV_low"], combined["NPV_high"])
            ],
            "Accuracy, %": combined["Accuracy"].map(format_percent),
            "Accuracy 95% CI, %": [
                format_ci(low, high)
                for low, high in zip(
                    combined["Accuracy_low"], combined["Accuracy_high"]
                )
            ],
            "F1, %": combined["F1"].map(format_percent),
        }
    )

    for column in ["Evaluable reports", "TP", "FP", "TN", "FN"]:
        out[column] = out[column].astype("Int64")

    return out


def format_heterogeneity_table(heterogeneity: pd.DataFrame) -> pd.DataFrame:
    """Format the manuscript Table 2-style heterogeneity output, with NE as needed."""
    out = heterogeneity[
        [
            "Analysis",
            "Model",
            "Metric",
            "n_sites",
            "total_n",
            "Q",
            "p_heterogeneity",
            "I2_percent",
        ]
    ].copy()

    analysis_order = ["PCI identification", "Complex PCI classification"]
    out["Analysis"] = pd.Categorical(
        out["Analysis"], categories=analysis_order, ordered=True
    )
    out = out.sort_values(["Analysis", "Model", "Metric"]).copy()

    out["Q statistic"] = out["Q"].map(
        lambda x: "NE" if pd.isna(x) else f"{float(x):.2f}"
    )
    out["P value"] = out["p_heterogeneity"].map(
        lambda x: "NE" if pd.isna(x) else ("<0.0001" if float(x) < 0.0001 else f"{float(x):.4f}")
    )
    out["I2-style statistic, %"] = out["I2_percent"].map(
        lambda x: "NE" if pd.isna(x) else f"{float(x):.1f}"
    )

    out = out.rename(
        columns={
            "n_sites": "Total evaluable sites",
            "total_n": "Total evaluable reports with site",
        }
    )

    return out[
        [
            "Analysis",
            "Model",
            "Metric",
            "Total evaluable sites",
            "Total evaluable reports with site",
            "Q statistic",
            "P value",
            "I2-style statistic, %",
        ]
    ]


def save_table(df: pd.DataFrame, outdir: Path, filename: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    df.to_csv(path, index=False)
    print(f"Saved table: {path}")


def save_all_tables(
    results: Mapping[str, pd.DataFrame],
    outdir: Path,
    model_order: Sequence[str],
) -> None:
    save_table(
        results["cohort"].sort_values("Model"),
        outdir,
        "cohort_and_evaluable_counts.csv",
    )
    save_table(
        results["pci_overall"].sort_values("Model"),
        outdir,
        "source_pci_identification_overall.csv",
    )
    save_table(
        results["pci_by_site"].sort_values([SITE_COL, "Model"]),
        outdir,
        "source_pci_identification_by_site.csv",
    )
    save_table(
        results["extraction_accuracy"].sort_values(["Variable", "Model"]),
        outdir,
        "source_extraction_accuracy_six_variables.csv",
    )
    save_table(
        results["complex_overall"].sort_values("Model"),
        outdir,
        "source_complex_pci_classification_overall.csv",
    )
    save_table(
        results["complex_by_site"].sort_values([SITE_COL, "Model"]),
        outdir,
        "source_complex_pci_classification_by_site.csv",
    )
    save_table(
        results["component_metrics"].sort_values(["Component", "Model"]),
        outdir,
        "source_complex_pci_criterion_metrics.csv",
    )
    save_table(
        results["heterogeneity"].sort_values(["Analysis", "Model", "Metric"]),
        outdir,
        "source_site_heterogeneity_numeric.csv",
    )
    save_table(
        results["heterogeneity_site_kn"].sort_values(
            ["Analysis", "Model", "Metric", SITE_COL]
        ),
        outdir,
        "source_site_heterogeneity_site_kn.csv",
    )

    supplementary = make_supplementary_metrics_table(
        results["pci_overall"],
        results["complex_overall"],
        results["component_metrics"],
        model_order,
    )
    save_table(
        supplementary,
        outdir,
        "supp_table1_full_metrics_pci_complex_pci_and_six_criteria.csv",
    )

    heterogeneity_table = format_heterogeneity_table(results["heterogeneity"])
    save_table(
        heterogeneity_table,
        outdir,
        "table2_site_heterogeneity_pci_and_complex_pci.csv",
    )


# =============================================================================
# MANUSCRIPT FIGURES
# =============================================================================

def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_marker_map(model_order: Sequence[str]) -> Dict[str, str]:
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
    return {
        model: marker_cycle[i % len(marker_cycle)] for i, model in enumerate(model_order)
    }


def save_figure(fig: plt.Figure, outdir: Path, basename: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    png_path = outdir / f"{basename}.png"
    pdf_path = outdir / f"{basename}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")


def add_vertical_performance_bands(ax: plt.Axes) -> None:
    edges = np.linspace(0, 1, 6)
    for i in range(len(edges) - 1):
        if i % 2 == 0:
            ax.axvspan(
                edges[i], edges[i + 1], facecolor="0.93", edgecolor="none", zorder=0
            )
    ax.set_xlim(0, 1.02)
    ax.set_xticks(edges)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="x", alpha=0.28, zorder=1)


def ci_xerr(row: pd.Series, metric: str) -> Tuple[float, float]:
    value = row.get(metric, np.nan)
    low = row.get(f"{metric}_low", np.nan)
    high = row.get(f"{metric}_high", np.nan)
    if pd.isna(value) or pd.isna(low) or pd.isna(high):
        return 0.0, 0.0
    return max(0.0, value - low), max(0.0, high - value)


def plot_binary_metrics_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    model_order: Sequence[str],
    markers: Mapping[str, str],
    panel_label: Optional[str] = None,
) -> None:
    y_base = np.arange(len(PERFORMANCE_METRICS))
    offsets = np.linspace(-0.22, 0.22, len(model_order))

    for offset, model_name in zip(offsets, model_order):
        sub = df[df["Model"] == model_name]
        if sub.empty:
            continue
        row = sub.iloc[0]

        values: List[float] = []
        low_err: List[float] = []
        high_err: List[float] = []

        for metric in PERFORMANCE_METRICS:
            values.append(row.get(metric, np.nan))
            if metric == "F1":
                low_err.append(0.0)
                high_err.append(0.0)
            else:
                low, high = ci_xerr(row, metric)
                low_err.append(low)
                high_err.append(high)

        ax.errorbar(
            values,
            y_base + offset,
            xerr=[low_err, high_err],
            fmt=markers[model_name],
            capsize=3,
            label=model_name,
            zorder=3,
        )

    display_title = f"{panel_label}. {title}" if panel_label else title
    ax.set_title(display_title, loc="left", pad=10)
    ax.set_yticks(y_base)
    ax.set_yticklabels(PERFORMANCE_METRICS)
    ax.set_ylim(len(PERFORMANCE_METRICS) - 0.5, -0.5)
    ax.set_xlabel("Performance")
    add_vertical_performance_bands(ax)


def plot_site_accuracy_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    model_order: Sequence[str],
    markers: Mapping[str, str],
    panel_label: Optional[str] = None,
) -> None:
    d = drop_missing_sites(df)
    sites = sorted(d[SITE_COL].dropna().unique()) if not d.empty else []

    display_title = f"{panel_label}. {title}" if panel_label else title
    ax.set_title(display_title, loc="left", pad=10)

    if not sites:
        ax.text(
            0.5,
            0.5,
            "No non-missing clinical sites available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    y_base = np.arange(len(sites))
    offsets = np.linspace(-0.22, 0.22, len(model_order))

    for offset, model_name in zip(offsets, model_order):
        sub = d[d["Model"] == model_name].set_index(SITE_COL).reindex(sites)
        values = sub["Accuracy"].astype(float).to_numpy()
        lows = sub["Accuracy_low"].astype(float).to_numpy()
        highs = sub["Accuracy_high"].astype(float).to_numpy()
        xerr = np.vstack(
            [np.clip(values - lows, 0, None), np.clip(highs - values, 0, None)]
        )

        ax.errorbar(
            values,
            y_base + offset,
            xerr=xerr,
            fmt=markers[model_name],
            capsize=3,
            label=model_name,
            zorder=3,
        )

    ax.set_yticks(y_base)
    ax.set_yticklabels(sites)
    ax.set_ylim(len(sites) - 0.5, -0.5)
    ax.set_xlabel("Accuracy")
    add_vertical_performance_bands(ax)


def add_shared_model_legend(
    fig: plt.Figure,
    axes: Sequence[plt.Axes],
    model_order: Sequence[str],
    y_anchor: float = 0.01,
) -> None:
    handles: List[Any] = []
    labels: List[str] = []
    for ax in np.atleast_1d(axes).ravel():
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break

    if handles:
        fig.legend(
            handles,
            labels,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, y_anchor),
            ncol=len(model_order),
            title="Model",
        )


def plot_main_figures(
    results: Mapping[str, pd.DataFrame],
    outdir: Path,
    model_order: Sequence[str],
) -> None:
    configure_plot_style()
    markers = make_marker_map(model_order)

    # Main Figure 2: PCI identification overall and by site.
    n_sites = results["pci_by_site"][SITE_COL].dropna().nunique()
    fig_height = max(5.8, 0.52 * max(n_sites, len(PERFORMANCE_METRICS)) + 2.2)
    fig, axes = plt.subplots(
        1, 2, figsize=(14.8, fig_height), gridspec_kw={"width_ratios": [1.0, 1.12]}
    )
    plot_binary_metrics_panel(
        axes[0],
        results["pci_overall"],
        "Overall PCI identification performance",
        model_order,
        markers,
        panel_label="A",
    )
    plot_site_accuracy_panel(
        axes[1],
        results["pci_by_site"],
        "PCI identification accuracy by clinical site",
        model_order,
        markers,
        panel_label="B",
    )
    add_shared_model_legend(fig, axes, model_order, y_anchor=0.005)
    fig.tight_layout(rect=[0, 0.065, 1, 1])
    save_figure(fig, outdir, "main_figure2_pci_identification_overall_and_by_site")
    plt.close(fig)

    # Main Figure 3: exact-match extraction accuracy.
    variables = [EXTRACTION_LABELS[c] for c in COMPONENT_ORDER]
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    y_base = np.arange(len(variables))
    offsets = np.linspace(-0.22, 0.22, len(model_order))

    for offset, model_name in zip(offsets, model_order):
        sub = (
            results["extraction_accuracy"]
            [results["extraction_accuracy"]["Model"] == model_name]
            .set_index("Variable")
            .reindex(variables)
        )
        values = sub["Accuracy"].astype(float).to_numpy()
        lows = sub["Accuracy_low"].astype(float).to_numpy()
        highs = sub["Accuracy_high"].astype(float).to_numpy()
        xerr = np.vstack(
            [np.clip(values - lows, 0, None), np.clip(highs - values, 0, None)]
        )
        ax.errorbar(
            values,
            y_base + offset,
            xerr=xerr,
            fmt=markers[model_name],
            capsize=3,
            label=model_name,
            zorder=3,
        )

    ax.set_title("Extraction accuracy for six complex PCI variables", pad=10)
    ax.set_yticks(y_base)
    ax.set_yticklabels(variables)
    ax.set_ylim(len(variables) - 0.5, -0.5)
    ax.set_xlabel("Exact-match accuracy")
    add_vertical_performance_bands(ax)
    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        title="Model",
    )
    fig.tight_layout()
    save_figure(fig, outdir, "main_figure3_extraction_accuracy_six_variables")
    plt.close(fig)

    # Main Figure 4: complex PCI overall and by site.
    n_sites = results["complex_by_site"][SITE_COL].dropna().nunique()
    fig_height = max(5.8, 0.52 * max(n_sites, len(PERFORMANCE_METRICS)) + 2.2)
    fig, axes = plt.subplots(
        1, 2, figsize=(14.8, fig_height), gridspec_kw={"width_ratios": [1.0, 1.12]}
    )
    plot_binary_metrics_panel(
        axes[0],
        results["complex_overall"],
        "Overall complex PCI classification performance",
        model_order,
        markers,
        panel_label="A",
    )
    plot_site_accuracy_panel(
        axes[1],
        results["complex_by_site"],
        "Complex PCI classification accuracy by clinical site",
        model_order,
        markers,
        panel_label="B",
    )
    add_shared_model_legend(fig, axes, model_order, y_anchor=0.005)
    fig.tight_layout(rect=[0, 0.065, 1, 1])
    save_figure(fig, outdir, "main_figure4_complex_pci_overall_and_by_site")
    plt.close(fig)

    # Supplementary Figure 1: TP/FP/TN/FN counts.
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.8), sharey=False)
    task_frames = [
        ("A", "PCI identification", results["pci_overall"]),
        ("B", "Complex PCI classification", results["complex_overall"]),
    ]
    bar_width = 0.18
    x = np.arange(len(model_order))
    offsets = np.linspace(-1.5 * bar_width, 1.5 * bar_width, 4)

    for ax, (panel_label, task_title, data) in zip(axes, task_frames):
        indexed = data.set_index("Model").reindex(model_order)
        for offset, count_name in zip(offsets, CONFUSION_COUNT_ORDER):
            values = indexed[count_name].astype(float).to_numpy()
            bars = ax.bar(x + offset, values, width=bar_width, label=count_name)
            for bar, value in zip(bars, values):
                if pd.notna(value):
                    ax.annotate(
                        f"{int(value)}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
        ax.set_title(f"{panel_label}. {task_title}", loc="left", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(model_order, rotation=18, ha="right")
        ax.set_ylabel("Number of reports")
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        title="Confusion-matrix count",
    )
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    save_figure(fig, outdir, "supp_figure1_tp_fp_tn_fn_pci_and_complex_pci")
    plt.close(fig)

    # Supplementary Figure 2: six criterion-level panels.
    fig, axes = plt.subplots(3, 2, figsize=(14.2, 15.4), sharex=False, sharey=False)
    for ax, panel_label, component in zip(axes.ravel(), list("ABCDEF"), COMPONENT_ORDER):
        sub = results["component_metrics"][
            results["component_metrics"]["Component"] == component
        ]
        plot_binary_metrics_panel(
            ax,
            sub,
            COMPLEX_COMPONENTS[component]["label"],
            model_order,
            markers,
            panel_label=panel_label,
        )
    add_shared_model_legend(fig, axes, model_order, y_anchor=0.015)
    fig.suptitle(
        "Performance for identification of each complex PCI criterion",
        y=0.995,
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.055, 1, 0.98])
    save_figure(
        fig,
        outdir,
        "supp_figure2_six_complex_pci_criterion_performance_panels",
    )
    plt.close(fig)


# =============================================================================
# COMMAND-LINE ENTRY POINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse LLM JSON outputs, merge them with a private adjudicated gold standard, "
            "and reproduce the complex-PCI manuscript evaluation."
        )
    )
    parser.add_argument(
        "--gold-standard",
        type=Path,
        required=True,
        help="Path to the final adjudicated gold-standard CSV (not distributed publicly).",
    )
    parser.add_argument(
        "--model",
        action="append",
        nargs=3,
        required=True,
        metavar=("NAME", "PCI_JSON", "EXTRACTION_JSON"),
        help=(
            "Model name followed by its PCI-identification JSON and complex-PCI extraction "
            "JSON. Repeat --model for each model."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results"),
        help="Directory for evaluation tables and figures (default: results).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Generate CSV results only; do not create PDF/PNG figures.",
    )
    parser.add_argument(
        "--save-merged",
        action="store_true",
        help=(
            "Also save each model's merged gold+prediction evaluation DataFrame locally. "
            "These files may contain protected study data and should not be committed."
        ),
    )
    return parser.parse_args()


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("_")


def main() -> None:
    args = parse_args()
    gold = load_gold_standard(args.gold_standard)

    model_order: List[str] = []
    model_dfs: Dict[str, pd.DataFrame] = {}

    for model_name, pci_json_str, extraction_json_str in args.model:
        if model_name in model_dfs:
            raise ValueError(f"Duplicate model name supplied: {model_name}")

        model_order.append(model_name)
        pci_json = Path(pci_json_str)
        extraction_json = Path(extraction_json_str)

        print(f"\nPreparing {model_name}...")
        df = prepare_model_dataframe(
            gold=gold,
            pci_json=pci_json,
            extraction_json=extraction_json,
            model_name=model_name,
        )
        model_dfs[model_name] = df

        print(
            f"  Gold rows: {len(df)} | "
            f"PCI predictions available: {df[PCI_PRED_COL].notna().sum()} | "
            f"Sites represented: {df[SITE_COL].dropna().nunique()}"
        )

        if args.save_merged:
            derived_dir = args.outdir / "private_derived_data"
            derived_dir.mkdir(parents=True, exist_ok=True)
            path = derived_dir / f"{safe_filename(model_name)}_gold_with_predictions.csv"
            df.to_csv(path, index=False)
            print(f"  Saved private merged data: {path}")

    results = evaluate_models(model_dfs, model_order)
    args.outdir.mkdir(parents=True, exist_ok=True)
    save_all_tables(results, args.outdir, model_order)

    if not args.skip_plots:
        plot_main_figures(results, args.outdir, model_order)

    print("\nEvaluation complete.")
    print(f"Results saved to: {args.outdir.resolve()}")

if __name__ == "__main__":
    main()
