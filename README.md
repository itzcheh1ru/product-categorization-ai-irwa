## Product-Categorization-AI

### Overview
Multi-agent AI system for product categorization and search, integrating LLMs, NLP, IR, and security.

### Components
- Backend: FastAPI (`backend/api/main.py`) with agent endpoints and AI search
- Agents: classifier, attribute extractor, tag generator
- Core: LLM and NLP helpers, security utils
- Frontend: Streamlit app (`frontend/app.py`)

### Run locally
```
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt -r frontend/requirements.txt scikit-learn
python run_app.py
```
- Backend: http://127.0.0.1:8000 (docs at /docs)
- Frontend: http://localhost:8501

### API
- Health: `GET /api/health`
- AI Search: `GET /api/search/suggest?q=red%20dress&top_n=5`
- Classifier: `POST /api/classifier/process`
- Extractor: `POST /api/extractor/process`
- Tagger: `POST /api/tagger/process`

### Responsible AI (summary)
- Input sanitization, logging hooks, transparent schemas, configurable models.

### Commercialization (summary)
- SaaS tiered pricing (Starter/Pro), target e-commerce SMEs, deploy via Docker.

### Contributors
- Group members per branch names.


