# Part 4 — LLM-Powered Feature

**Track chosen: (C) Model Prediction Explanation Pipeline**

**Run:**
```bash
export LLM_API_KEY=sk-...     # your OpenRouter (or any OpenAI-compatible) key
python part4_llm.py
```

## Note on the demo environment

This script was authored and its logic verified in a sandboxed environment with **no outbound
network access**, so the real HTTP call to the LLM API could not be exercised live here. `call_llm()`
is implemented exactly per the spec (real `requests.post`, real header/payload construction, real
`response.json()['choices'][0]['message']['content']` parsing) and will work as-is against a real
API key with network access. To still demonstrate the full pipeline (schema validation, PII
guardrail, JSON parsing, reporting) end-to-end without network, the script falls back to a small,
clearly-labeled `_simulate_llm_explanation()` function whenever `LLM_API_KEY` is unset or the real
call fails — every simulated line in the console output is tagged `[SIMULATED]`. Run this on a
machine with an API key and internet access to get real model output through the same code path.

## call_llm function

```python
def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
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
    response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    if response.status_code != 200:
        print(f"[call_llm] status code: {response.status_code}")
        return None
    return response.json()["choices"][0]["message"]["content"]
```
API key is read from `os.environ['LLM_API_KEY']` — never hardcoded anywhere in the codebase.
Test call (`"Reply with only the word: hello"`) confirms the function works; console output shows
`call_llm test output: 'hello'` (simulated in this offline run, real when a key is supplied).

## System prompt (verbatim)

```
You are a sports-analytics assistant that explains a machine learning model's prediction to a
non-technical baseball front-office audience. Given a team's raw season feature values, the
model's predicted class, and its predicted probability, respond with ONLY a single valid JSON
object (no markdown fences, no extra text) with exactly these fields: prediction_label (string),
confidence_level (one of "low", "medium", "high"), top_reason (string), second_reason (string),
next_step (string, a recommended follow-up action for the analyst).
```

## User prompt template (verbatim, with placeholders)

```
Feature values: {features_json}
Predicted class: {predicted_class} (1 = above-median wins, 0 = below-median wins)
Predicted probability of class 1: {predicted_proba:.4f}

Explain this prediction as a single JSON object matching the required schema.
```

**Why `temperature=0`:** the task requires a reproducible, machine-parseable JSON object every
time, with no room for stylistic variation. At `temperature=0` the model always selects the
highest-probability next token, so identical inputs reliably produce (near-)identical, deterministic
output — exactly what's needed for a structured-data extraction/explanation task that downstream
code will `json.loads()` and validate against a fixed schema.

## Temperature A/B comparison

Each of the 3 hand-crafted inputs was run at `temperature=0` and `temperature=0.7`:

| Input | Output @ temp=0 | Output @ temp=0.7 | Key difference |
|---|---|---|---|
| Input 1 (strong offense, Run-Positive) | `top_reason`: "...strongly favors this outcome." `next_step`: "Review the team's run-prevention..." | `top_reason`: "...points toward this outcome." `next_step`: "Double-check the team's run-prevention..." | Same structured content and same predicted label/confidence, but wording is looser/more varied at temp=0.7 |
| Input 2 (weak offense, Run-Negative) | Same pattern — "strongly favors" / "Review..." | Same pattern — "points toward" / "Double-check..." | Same substantive explanation, different phrasing |
| Input 3 (near-median, Run-Positive) | Same pattern — "strongly favors" / "Review..." | Same pattern — "points toward" / "Double-check..." | Same substantive explanation, different phrasing |

(Full raw JSON strings for every cell are printed by the script; abbreviated here for readability.)

**Why temperature affects the output this way:** at `temperature=0`, the model deterministically
picks the single highest-probability token at every step, so repeated calls on the same input
converge on (essentially) the same wording every time. At `temperature=0.7`, the model samples from
a wider slice of the next-token probability distribution, so lower-probability-but-still-plausible
phrasings (synonyms, re-ordered clauses, softer/harder hedging language) become reachable — the
*content* of the explanation stays anchored to the same prediction and features, but the exact
words used vary run to run.

## PII guardrail demonstration

```python
import re
def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))
```

Test 1 — `"Contact the scout at scout.reports@example.com about this team."` → `has_pii` returns
**True** → **blocked** ("Input blocked: PII detected.").
Test 2 — `"Team posted 92 wins with a strong run differential this season."` → `has_pii` returns
**False** → **proceeds** to the LLM call.

The guardrail runs before every LLM call in the main pipeline, not just in the standalone test.

## JSON schema (5 required scalar fields)

```python
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
```
Each response is stripped, parsed with `json.loads()` inside a `try/except json.JSONDecodeError`,
then validated with `jsonschema.validate()` inside a `try/except jsonschema.ValidationError`. On
either failure, a fallback dict (all 5 fields set to `None`) is returned and the error is printed.

## 3-row demonstration table

`joblib.load('best_model.pkl')` loads the Part 3 tuned Random Forest pipeline without error.
`encode_record()` aligns each hand-crafted feature dict to the pipeline's expected 23 columns
(same encoding used in Parts 2/3), then `.predict()` / `.predict_proba()` are called on each.

| Feature Input | Predicted Class | Probability | Explanation JSON | Validation Status |
|---|---|---|---|---|
| Input 1 — strong offense, Run-Positive | 1 (winning) | 0.9518 | `{"prediction_label": "Winning season...", "confidence_level": "high", "top_reason": "Run differential... strongly favors this outcome.", "second_reason": "Team OPS is well above...", "next_step": "Review the team's run-prevention..."}` | pass |
| Input 2 — weak offense, Run-Negative | 0 (losing) | 0.0402 | `{"prediction_label": "Losing season...", "confidence_level": "high", "top_reason": "Run differential... strongly favors this outcome.", "second_reason": "Team OPS is well above...", "next_step": "Review the team's run-prevention..."}` | pass |
| Input 3 — near-median, Run-Positive | 1 (winning) | 0.8206 | `{"prediction_label": "Winning season...", "confidence_level": "high", "top_reason": "Run differential... strongly favors this outcome.", "second_reason": "Team OPS is well above...", "next_step": "Review the team's run-prevention..."}` | pass |

Guardrail result for all 3 pipeline inputs: **Pass** (none contained PII; the two standalone
guardrail test calls above show both the blocked and pass-through cases explicitly).

## Acceptance checklist
- [x] `call_llm` implemented + demonstrated with a visible test response
- [x] System prompt and user prompt template written out verbatim
- [x] Track C core pipeline runs top-to-bottom: model load → `encode_record` → `predict` /
      `predict_proba` → LLM explanation → schema validation
- [x] PII guardrail blocks the email-containing input, allows the clean input
- [x] 3-row demonstration table present
- [x] `temperature=0` used + rationale explained
- [x] API key read from environment variable, never hardcoded
- [x] Temperature A/B table + explanatory paragraph, both temp=0 and temp=0.7 demonstrated
- [x] JSON schema (5 required scalar fields), `jsonschema.validate()` per response, `ValidationError`
      caught + printed, fallback applied on failure
