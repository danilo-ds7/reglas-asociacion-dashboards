"""
Publica las dos apps como Spaces de Hugging Face (SDK Docker).

Uso:
    pip install -U huggingface_hub
    huggingface-cli login            # pega tu token con permiso "write"
    python publicar_en_hf.py TU_USUARIO
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

usuario = sys.argv[1] if len(sys.argv) > 1 else input("Usuario u organización de Hugging Face: ").strip()
api = HfApi()
base = Path(__file__).parent

for carpeta, nombre in [("streamlit_app", "reglas-asociacion-streamlit"),
                        ("shiny_app", "reglas-asociacion-shiny")]:
    repo = f"{usuario}/{nombre}"
    api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=base / carpeta, repo_id=repo, repo_type="space",
                      ignore_patterns=["__pycache__/*", "*.pyc"])
    print(f"✅ https://huggingface.co/spaces/{repo}")
