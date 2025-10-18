# 🛠️ Copilot Instructions for Product-Categorization-AI

## Big Picture Architecture
- **Multi-agent AI system** for product categorization, attribute extraction, tag generation, and smart search.
- **Backend**: FastAPI (`backend/api/main.py`) exposes agent endpoints and AI search.
- **Agents**: Each agent (category classifier, attribute extractor, tag generator, orchestrator) is in `backend/agents/` and called via API routes.
- **Core**: LLM integration (`llm_integration.py`), NLP helpers, security (JWT, sanitization, CSRF) in `backend/core/`.
- **Frontend**: Streamlit UI (`frontend/app.py`) interacts with backend via HTTP.
- **Data**: Product data in CSV/JSON under `backend/data/`.

## Developer Workflows
- **Setup**: Use Python 3.12+ for best compatibility and speed.
- **Install**: `pip install -r backend/requirements.txt -r frontend/requirements.txt scikit-learn`
- **Run**: `python run_app.py` (starts both backend and frontend)
- **Direct Backend**: `uvicorn backend/api/main:app --host 0.0.0.0 --port 8000`
- **Direct Frontend**: `streamlit run frontend/app.py --server.port 8501`
- **Tests**: Run test files in project root and `backend/tests/` (e.g., `pytest backend/tests/`)

## Project-Specific Patterns
- **Agents**: Each agent is a class in `backend/agents/`, called by orchestrator or directly via API.
- **API Security**: Optional API key via `API_KEY` env var and `X-API-Key` header.
- **LLM Model**: Configurable via `LLM_MODEL` env var; default is `llama3.1`.
- **Data Flow**: Product description → NLP preprocessing → LLM agent(s) → IR search → response.
- **Responsible AI**: Input sanitization, confidence scores, transparent schemas, logging (see `RESPONSIBLE_AI.md`).

## Integration Points
- **LLM**: Uses Ollama wrapper in `backend/core/llm_integration.py`.
- **IR/Search**: TF-IDF similarity in `backend/core/information_retrieval.py`.
- **Security**: JWT, sanitization, CSRF in `backend/core/security.py`.
- **Frontend/Backend**: Communicate via HTTP endpoints (see API docs in README).

## Key Files & Directories
- `backend/api/main.py`, `routes.py`: FastAPI app and endpoints
- `backend/agents/`: All agent classes
- `backend/core/`: LLM, NLP, security, IR logic
- `frontend/app.py`: Streamlit UI
- `backend/data/`: Product data sources
- `RESPONSIBLE_AI.md`, `EVALUATION.md`, `COMMERCIALIZATION.md`: Policy, evaluation, and business docs

## Examples
- To add a new agent: create a class in `backend/agents/`, add endpoint in `backend/api/routes.py`, update orchestrator if needed.
- To change LLM model: set `LLM_MODEL` in `.env` or environment.
- To secure API: set `API_KEY` and require `X-API-Key` header.

## Conventions
- Use explicit imports and type hints in Python code.
- Keep agent logic isolated; orchestrator coordinates multi-agent flows.
- Data files should be placed in `backend/data/`.
- All environment variables documented in README.

---
For more details, see the project README and referenced docs. If any section is unclear or missing, please provide feedback to improve these instructions.
