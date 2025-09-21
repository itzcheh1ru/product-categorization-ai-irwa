from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.utils.config_loader import load_config
from backend.core.information_retrieval import InformationRetrieval
from backend.agents.orchestractor_agent import OrchestratorAgent
from .routes import router as api_router  # make sure routes.py uses OrchestratorAgent properly

import logging

# -----------------------------
# Configure logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Load configuration
# -----------------------------
config = load_config()

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI(
    title="Product Categorization AI API",
    description="API for AI-powered product categorization, attribute extraction, and tag generation",
    version=config.get('app', {}).get('version', '1.0')
)

# -----------------------------
# Add CORS middleware
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Initialize Information Retrieval
# -----------------------------
ir = InformationRetrieval(data_path="backend/data/cleaned_product_data.csv")

# -----------------------------
# Initialize Orchestrator Agent
# -----------------------------
orchestrator = OrchestratorAgent(ir_instance=ir)

# -----------------------------
# Include API routes
# -----------------------------
app.include_router(api_router, prefix="/api")

# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
async def root():
    return {
        "message": "Product Categorization AI API",
        "version": config.get('app', {}).get('version', '1.0'),
        "docs": "/docs"
    }

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.get('api', {}).get('host', '127.0.0.1'),
        port=config.get('api', {}).get('port', 8000),
        reload=config.get('api', {}).get('debug', True)
    )
