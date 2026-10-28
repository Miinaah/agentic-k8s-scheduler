"""
backend/run.py

Entrypoint script to start the FastAPI server with uvicorn.
Usage: python backend/run.py [--port 8000] [--host 0.0.0.0]
"""

import argparse
import sys
from pathlib import Path

# Add backend directory and root packages to sys.path
backend_dir = Path(__file__).resolve().parent
root_dir = backend_dir.parent
venv_packages = root_dir / ".venv" / "Lib" / "site-packages"

if venv_packages.is_dir() and str(venv_packages) not in sys.path:
    sys.path.insert(0, str(venv_packages))
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(root_dir / "split-c") not in sys.path:
    sys.path.insert(0, str(root_dir / "split-c"))
if str(root_dir / "split-ab") not in sys.path:
    sys.path.insert(0, str(root_dir / "split-ab"))

import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the Agentic K8s Scheduler API server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload")
    args = parser.parse_args()

    print(f"Starting Agentic K8s Scheduler API on http://localhost:{args.port} ...")
    uvicorn.run("app:app", host=args.host, port=args.port, reload=args.reload)
