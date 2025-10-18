import subprocess
import sys
import os

def run_backend():
    print("🚀 Starting FastAPI backend...")
    os.chdir("backend")
    subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--reload", "--port", "8000"])

def run_frontend():
    print("🎨 Starting Streamlit frontend...")
    os.chdir("../frontend")
    subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    run_backend()
    run_frontend()
    print("✅ Backend running on http://127.0.0.1:8000  |  Frontend running on http://localhost:8501")
