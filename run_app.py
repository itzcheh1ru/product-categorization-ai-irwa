import subprocess
import sys
import os

def run_backend():
    print("🚀 Starting FastAPI backend...")
    os.chdir("backend")
    # Bind to 0.0.0.0 for Railway/container environments
    subprocess.Popen([
        sys.executable, "-m", "uvicorn", "api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000"
    ])

def run_frontend():
    print("🎨 Starting Streamlit frontend...")
    os.chdir("../frontend")
    # Streamlit must expose the public $PORT in Railway; default to 8501 locally
    port = os.getenv("PORT", os.getenv("STREAMLIT_SERVER_PORT", "8501"))
    subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.address", "0.0.0.0",
        "--server.port", str(port),
        "--server.headless", "true"
    ])

if __name__ == "__main__":
    run_backend()
    run_frontend()
    print("✅ Backend running on http://127.0.0.1:8000  |  Frontend running on http://0.0.0.0:" + os.getenv("PORT", "8501"))
