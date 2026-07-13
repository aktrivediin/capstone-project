"""
Part 4 -- LLM-Powered Feature
Track chosen: (C) Model Prediction Explanation Pipeline

Loads best_model.pkl from Part 3, runs 3 hand-crafted feature vectors through
.predict()/.predict_proba(), and asks an LLM to explain each prediction as
validated structured JSON.

Run:
    export LLM_API_KEY=sk-...          # your OpenRouter (or compatible) key
    python part4_llm.py

If LLM_API_KEY is not set, or the API is unreachable, the script falls back to
a clearly-labeled local SIMULATED response generator so the rest of the
pipeline (schema validation, guardrails, reporting) can still be demonstrated
end-to-end offline. See README.md, section "Note on the demo environment".
"""

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import requests
import jsonschema

# ---------------------------------------------------------------------------
# LLM API configuration
# ---------------------------------------------------------------------------
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-4o-mini"   # any OpenAI-compatible chat model works here
API_KEY = os.environ.get("LLM_API_KEY")


def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    """Reusable LLM call. Returns the assistant's text content, or None on failure."""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    except requests.exceptions.RequestException as e:
        print(f"[call_llm] network error: {e}")
        return None
    if response.status_code != 200:
        print(f"[call_llm] status code: {response.status_code}")
        return None
    return response.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Demo-environment fallback: this sandbox has no outbound network access, so
# a clearly-labeled SIMULATED responder stands in for call_llm() when the
# real call is unavailable, purely so the schema-validation / guardrail /
# reporting pipeline below can be demonstrated end-to-end. Every simulated
# response is printed with a "[SIMULATED]" tag. With LLM_API_KEY set and
# network access, real_call_or_simulated() calls the real call_llm() above
# and this fallback path is never used.
# ---------------------------------------------------------------------------
def _simulate_llm_explanation(predicted_class, predicted_proba):
    label = "Winning season (above median wins)" if predicted_class == 1 else "Losing season (below median wins)"
    conf = "high" if abs(predicted_proba - 0.5) > 0.3 else ("medium" if abs(predicted_proba - 0.5) > 0.15 else "low")
    body = {
        "prediction_label": label,
        "confidence_level": conf,
        "top_reason": "Run differential (runs scored minus runs allowed) strongly favors this outcome.",
        "second_reason": "Team OPS is well above/below the league-average band seen in training.",
        "next_step": "Review the team's run-prevention (pitching/defense) metrics for confirmation.",
    }
    return json.dumps(body)


def real_call_or_simulated(system_prompt, user_prompt, temperature):
    if API_KEY:
        result = call_llm(system_prompt, user_prompt, temperature=temperature)
        if result is not None:
            return result, False
        print("[part4] Real API call failed/unavailable -> falling back to SIMULATED response.")
    else:
        print("[part4] LLM_API_KEY not set -> using SIMULATED response for this offline demo run.")
    return None, True


def section(t):
    print("\n" + "=" * 90)
    print(t)
    print("=" * 90)


# ---------------------------------------------------------------------------
# Task: test call_llm with a simple prompt
# ---------------------------------------------------------------------------
section("Test call: simple deterministic prompt")
test_out, was_simulated = real_call_or_simulated(
    system_prompt="You reply with exactly one word and nothing else.",
    user_prompt="Reply with only the word: hello",
    temperature=0.0,
)
if was_simulated:
    test_out = "hello"  # deterministic simulated stand-in for the smoke test
print(f"call_llm test output: {test_out!r}  (simulated={was_simulated})")

# ---------------------------------------------------------------------------
# PII guardrail
# ---------------------------------------------------------------------------
def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))


section("Guardrail demonstration")
pii_input = "Contact the scout at scout.reports@example.com about this team."
clean_input = "Team posted 92 wins with a strong run differential this season."
print("PII input blocked?:", has_pii(pii_input), "->", "Input blocked: PII detected." if has_pii(pii_input) else "proceeds")
print("Clean input blocked?:", has_pii(clean_input), "-> proceeds to LLM call" if not has_pii(clean_input) else "blocked")

# ---------------------------------------------------------------------------
# Load best model + build encode_record() aligned to Part 2/3 feature columns
# ---------------------------------------------------------------------------
section("Load best_model.pkl and prepare encode_record()")
model = joblib.load("../part3_ensembles/best_model.pkl")

reference_df = pd.read_csv("../part1_eda/cleaned_data.csv")
X_reference = reference_df.drop(columns=["W", "ID", "Win_Tier"])
X_reference = pd.get_dummies(X_reference, columns=["Run_Diff_Type"], drop_first=True)
FEATURE_COLUMNS = X_reference.columns.tolist()
print("Model expects", len(FEATURE_COLUMNS), "feature columns.")


def encode_record(features: dict) -> pd.DataFrame:
    """Align a hand-crafted feature dict (raw, unscaled) to the pipeline's
    expected column set/order. The pipeline itself (SimpleImputer +
    StandardScaler + RandomForestClassifier) handles imputation/scaling."""
    row = pd.DataFrame([features])
    if "Run_Diff_Type" in row.columns:
        row = pd.get_dummies(row, columns=["Run_Diff_Type"])
    row = row.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return row


# 3 hand-crafted feature-vector inputs (drawn from realistic value ranges seen in Part 1's describe())
hand_crafted_inputs = [
    {  # Strong offensive team, positive run diff
        "G": 162, "R": 850, "AB": 5550, "H": 1500, "1B": 1000, "2B": 290, "3B": 35, "HR": 190,
        "BB": 600, "SO": 950, "SB": 110, "CS": 45, "HBP": 40, "BBHBP": 640, "SF": 45,
        "Outs": 4080, "Outsinplay": 3100, "RA": 620, "BA": 0.270, "OBA": 0.335, "SLG": 0.430,
        "OPS": 0.765, "Run_Diff_Type": "Run-Positive",
    },
    {  # Weak offensive team, negative run diff
        "G": 162, "R": 600, "AB": 5450, "H": 1350, "1B": 950, "2B": 230, "3B": 25, "HR": 110,
        "BB": 460, "SO": 1050, "SB": 90, "CS": 50, "HBP": 15, "BBHBP": 475, "SF": 10,
        "Outs": 4100, "Outsinplay": 3050, "RA": 780, "BA": 0.248, "OBA": 0.310, "SLG": 0.370,
        "OPS": 0.680, "Run_Diff_Type": "Run-Negative",
    },
    {  # Borderline / near-median team
        "G": 162, "R": 710, "AB": 5520, "H": 1430, "1B": 985, "2B": 265, "3B": 30, "HR": 150,
        "BB": 520, "SO": 970, "SB": 100, "CS": 48, "HBP": 20, "BBHBP": 540, "SF": 15,
        "Outs": 4085, "Outsinplay": 3070, "RA": 710, "BA": 0.261, "OBA": 0.327, "SLG": 0.402,
        "OPS": 0.729, "Run_Diff_Type": "Run-Positive",
    },
]

# ---------------------------------------------------------------------------
# JSON schema (5+ required scalar fields)
# ---------------------------------------------------------------------------
EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string"},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"},
    },
    "required": ["prediction_label", "confidence_level", "top_reason", "second_reason", "next_step"],
}

FALLBACK = {k: None for k in EXPLANATION_SCHEMA["required"]}

SYSTEM_PROMPT = (
    "You are a sports-analytics assistant that explains a machine learning model's "
    "prediction to a non-technical baseball front-office audience. Given a team's raw "
    "season feature values, the model's predicted class, and its predicted probability, "
    "respond with ONLY a single valid JSON object (no markdown fences, no extra text) "
    "with exactly these fields: prediction_label (string), confidence_level "
    "(one of \"low\", \"medium\", \"high\"), top_reason (string), second_reason (string), "
    "next_step (string, a recommended follow-up action for the analyst)."
)

USER_PROMPT_TEMPLATE = (
    "Feature values: {features_json}\n"
    "Predicted class: {predicted_class} (1 = above-median wins, 0 = below-median wins)\n"
    "Predicted probability of class 1: {predicted_proba:.4f}\n\n"
    "Explain this prediction as a single JSON object matching the required schema."
)

# ---------------------------------------------------------------------------
# Temperature A/B comparison + main validated pipeline
# ---------------------------------------------------------------------------
section("Run pipeline on 3 hand-crafted inputs: predict -> LLM explanation -> validate")

demo_rows = []
temp_ab_rows = []

for i, features in enumerate(hand_crafted_inputs, start=1):
    encoded = encode_record(features)
    pred_class = int(model.predict(encoded)[0])
    pred_proba = float(model.predict_proba(encoded)[0, 1])
    print(f"\n--- Input {i} ---")
    print("Features:", features)
    print(f"Predicted class: {pred_class}   Predicted P(class=1): {pred_proba:.4f}")

    features_json = json.dumps(features)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        features_json=features_json, predicted_class=pred_class, predicted_proba=pred_proba
    )

    # Guardrail check before every LLM call
    if has_pii(user_prompt):
        print("Input blocked: PII detected.")
        demo_rows.append({"input": f"Input {i}", "llm_output": None, "valid_json": "N/A", "guardrail": "Blocked"})
        continue

    # --- temperature = 0 (main pipeline run) ---
    raw_response, simulated0 = real_call_or_simulated(SYSTEM_PROMPT, user_prompt, temperature=0.0)
    if simulated0:
        raw_response = _simulate_llm_explanation(pred_class, pred_proba)
    tag0 = "[SIMULATED] " if simulated0 else ""
    print(f"{tag0}LLM raw response (temp=0): {raw_response}")

    validation_status = "pass"
    try:
        cleaned = raw_response.strip()
        parsed = json.loads(cleaned)
        try:
            jsonschema.validate(parsed, EXPLANATION_SCHEMA)
        except jsonschema.ValidationError as ve:
            print("Schema validation error:", ve)
            validation_status = f"fail (schema: {ve})"
            parsed = FALLBACK
    except json.JSONDecodeError as je:
        print("JSON decode error:", je)
        validation_status = f"fail (json: {je})"
        parsed = FALLBACK

    print("Validation outcome:", validation_status)
    demo_rows.append({
        "input": f"Input {i}",
        "llm_output": parsed,
        "valid_json": validation_status,
        "guardrail": "Pass",
    })

    # --- temperature = 0.7 (A/B comparison) ---
    raw_response_07, simulated07 = real_call_or_simulated(SYSTEM_PROMPT, user_prompt, temperature=0.7)
    if simulated07:
        # simulate mild wording variability for the demo, still schema-valid
        alt = json.loads(_simulate_llm_explanation(pred_class, pred_proba))
        alt["top_reason"] = alt["top_reason"].replace("strongly favors", "points toward")
        alt["next_step"] = alt["next_step"].replace("Review", "Double-check")
        raw_response_07 = json.dumps(alt)
    tag07 = "[SIMULATED] " if simulated07 else ""
    print(f"{tag07}LLM raw response (temp=0.7): {raw_response_07}")

    temp_ab_rows.append({
        "input": f"Input {i}",
        "temp_0": raw_response,
        "temp_07": raw_response_07,
    })

# ---------------------------------------------------------------------------
# Print demonstration tables
# ---------------------------------------------------------------------------
section("3-row demonstration table")
for row in demo_rows:
    print(row)

section("Temperature A/B table")
for row in temp_ab_rows:
    print(row)

print("\nDONE.")
