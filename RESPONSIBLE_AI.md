# Responsible AI Practices

## Principles
- Fairness: avoid biased prompts; review datasets for representativeness.
- Transparency: document model versions and decisions.
- Privacy: sanitize inputs, avoid logging PII, use JWT for auth.
- Safety: validate outputs, set confidence thresholds.

## Implementation Hooks
- Input sanitization: `backend/core/security.py::sanitize_input`
- Confidence scores on tags/classification
- Logging placeholders for audits (avoid raw PII)
- Configurable model via env `LLM_MODEL`

## Evaluation
- Offline tests for precision@k on search suggestions
- Human review for attribute extraction correctness


