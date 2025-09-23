# 🤝 Responsible AI Practices

## 🧭 Principles
- ⚖️ Fairness: avoid biased prompts; review datasets for representativeness
- 🔎 Transparency: document model versions and key decisions
- 🔐 Privacy: sanitize inputs, avoid logging PII, use JWT/API keys
- 🛡️ Safety: validate outputs, use confidence thresholds and fallbacks

## 🛠️ Implementation Hooks
- 🧼 Input sanitization: `backend/core/security.py::sanitize_input`
- 📊 Confidence scores on tags/classification
- 🧾 Audit-friendly logging (avoid raw PII)
- ⚙️ Configurable model via env `LLM_MODEL`
- 🗝️ Optional API key with `X-API-Key` header

## 🔬 Evaluation
- 🔁 Offline tests for precision@k/recall@k on search suggestions
- 👀 Human review for attribute extraction correctness
- 🧪 Prompt iteration with example bank and regression checks

## 📝 Guidance for Report/Viva
- Explain how data is sanitized and stored (no PII logs)
- Show examples demonstrating fairness and mitigations
- Include model/version provenance and change logs


