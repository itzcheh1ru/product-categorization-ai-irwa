# 🎯 Hirusha's Components — Product Categorization AI

This document highlights the work by **Hirusha (IT23426580)** in the Product Categorization AI project.

---

## 🧩 Assigned Components

### 1) 🏷️ TagGeneratorAgent (`backend/agents/tag_generator_agent.py`)
- **Purpose**: Generate relevant, specific tags using LLM + NLP + rules
- **Features**
  - 🤖 LLM tags (structured output)
  - 🧠 NLP tags (nouns/adjectives, cleaned)
  - 📐 Rule tags (color/season/usage/article type)
  - ♻️ Dedup + sort by confidence, capped to top 10
- **Example**
```python
from agents.tag_generator_agent import TagGeneratorAgent
agent = TagGeneratorAgent()
product = {"productDisplayName": "Red Cotton Summer Dress", "baseColour": "Red"}
print(agent.generate_tags(product))
```

### 2) 🔐 Security Module (`backend/core/security.py`)
- **Purpose**: Auth & data protection utilities
- **Features**
  - 🔑 Password hashing/verify (bcrypt)
  - 🪪 JWT access/refresh tokens
  - 🧼 Input sanitization (XSS/SQL patterns)
  - 🛡️ CSRF tokens + verification
  - 🗂️ Safe filename checks, basic rate limiting
- **Example**
```python
from core.security import get_password_hash, verify_password
hashed = get_password_hash("SecurePassword123!")
print(verify_password("SecurePassword123!", hashed))
```

### 3) 📦 API Schemas (`backend/api/schemas.py`)
- **Purpose**: Pydantic models for requests/responses
- **Includes**
  - `ProductData`, `Tag`, `TagGenerationRequest/Response`
  - Validation (confidence bounds), helpful field descriptions

---

## 🧪 Testing (`test_hirusha_components.py`)
Run unit tests for tags, security, and schemas.
```bash
source venv/bin/activate
python test_hirusha_components.py
```
✅ All tests passed locally.

---

## 🗂️ File Structure (mine)
```
backend/
  agents/tag_generator_agent.py
  core/security.py
  api/schemas.py
```

---

## 🚀 How to Run (project)
```bash
source venv/bin/activate
python run_app.py
```
- API docs: http://127.0.0.1:8000/docs
- Streamlit UI: http://localhost:8501

---

## 📈 Next Ideas
- Persist generated tags to DB
- Tag analytics (top tags, coverage)
- Rate-limit per user / API key

---

## 📬 Contact
**Hirusha** — IT23426580
- Components: TagGeneratorAgent, Security, Schemas
- Status: ✅ Completed + Tested
