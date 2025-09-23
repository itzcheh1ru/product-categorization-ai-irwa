# 🛍️ Product-Categorization-AI

Multi‑agent AI system for product categorization, attribute extraction, tag generation, and smart search — integrating LLMs, NLP, IR, and security.

---

## ✨ What’s Inside
- ⚙️ Backend: FastAPI API (`backend/api/main.py`) with agent endpoints and AI search
- 🧠 Agents: Category Classifier, Attribute Extractor, Tag Generator
- 🧩 Core: LLM wrapper, NLP helpers, Security utils (JWT, sanitization, CSRF)
- 🎨 Frontend: Streamlit UI (`frontend/app.py`)
- 🔎 AI Search: TF‑IDF based similar product suggestions

---

## 🚀 Quick Start
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt -r frontend/requirements.txt scikit-learn
python run_app.py
```
- 🧪 Backend: http://127.0.0.1:8000 (OpenAPI docs at `/docs`)
- 🎛️ Frontend: http://localhost:8501

Tip: If scikit‑learn install is slow on macOS/Python 3.13, use Python 3.12 for prebuilt wheels.

---

## 🔐 Optional API Security
Set an API key to protect endpoints:
```bash
export API_KEY="your-secret-key"
```
Send the header on requests:
```
X-API-Key: your-secret-key
```

---

## 📡 API Endpoints
- ❤️ Health: `GET /api/health`
- 🔍 AI Search: `GET /api/search/suggest?q=red%20dress&top_n=5`
- 🧭 Classifier: `POST /api/classifier/process`
- 🧪 Extractor: `POST /api/extractor/process`
- 🏷️ Tagger: `POST /api/tagger/process`
- 🤖 Orchestrator (full pipeline): `POST /api/orchestrator/process`

Example body (POST endpoints):
```json
{
  "description": "Red cotton summer dress"
}
```

---

## 📂 Data
- Default search corpus: `backend/data/cleaned_product_data.csv`
- Fallback: `backend/data/product.csv`

---

## 🧠 How It Works (High Level)
1. ✍️ Input product description
2. 🧪 NLP preprocessing + (optional) NER/summarization
3. 🧠 LLM (via Ollama wrapper) for classification/attributes/tags
4. 🔎 IR: TF‑IDF similarity for product suggestions
5. 🔐 Security: sanitization, JWT helpers, optional API key

> LLM model configurable via env `LLM_MODEL` (e.g., `llama3.1`).

---

## 🧭 Project Structure
```
backend/
  api/
    main.py        # FastAPI app + AI search
    routes.py      # Agent endpoints
  agents/
    orchestrator_agent.py
    category_classifier_agent.py
    attribute_extractor_agent.py
    tag_generator_agent.py
  core/
    llm_integration.py
    nlp_processor.py
    security.py
frontend/
  app.py           # Streamlit UI
```

---

## ✅ Responsible AI (summary)
- 🧹 Input sanitization; confidence scores on outputs
- 🧾 Transparent schemas & logging hooks (avoid PII)
- ⚙️ Configurable model and thresholds

Read more: `RESPONSIBLE_AI.md`

---

## 💼 Commercialization (summary)
- 🎯 Target: e‑commerce SMEs
- 💰 Pricing tiers: Starter / Pro / Enterprise
- ☁️ Deployment: Dockerized FastAPI + Streamlit; optional managed hosting

Details: `COMMERCIALIZATION.md`

---

## 📊 Evaluation
- 🔎 Search: precision@k, recall@k
- 🏷️ Tag relevance: human ratings
- 🧭 Classification accuracy vs labeled set

Plan & results: `EVALUATION.md`

---

## 👥 Contributors
- List your team members and roles here.

---

## 🧰 Troubleshooting
- ⏳ Slow install on macOS/ARM? Use Python 3.12
- 🔑 401 from API? Set `API_KEY` and send `X-API-Key`
- 📦 Missing sklearn? `pip install scikit-learn`

Happy hacking! 🚀


