"""
deploy_to_hf.py
Uploads the model and Streamlit Space to Hugging Face Hub.

Prerequisites:
    hf auth login      # paste your HF write token when prompted

Then run:
    python deploy_to_hf.py
"""
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo

HF_USERNAME   = "Nyingi101"
MODEL_REPO_ID = f"{HF_USERNAME}/stunting-risk-scorer"
SPACE_REPO_ID = f"{HF_USERNAME}/stunting-risk-heatmap"

BASE    = Path(__file__).parent
SPACE_DIR = BASE / "hf_space"

api = HfApi()


def deploy_model():
    print(f"\n{'─'*55}")
    print(f"  Deploying MODEL → {MODEL_REPO_ID}")
    print(f"{'─'*55}")

    create_repo(MODEL_REPO_ID, repo_type="model",
                exist_ok=True, private=False)

    # Model card (README.md)
    api.upload_file(
        path_or_fileobj=str(BASE / "hf_model_card.md"),
        path_in_repo="README.md",
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        commit_message="Add model card",
    )
    print("  ✓ README.md (model card)")

    # Serialised model
    api.upload_file(
        path_or_fileobj=str(BASE / "output" / "scorer.pkl"),
        path_in_repo="scorer.pkl",
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        commit_message="Add scorer.pkl",
    )
    print("  ✓ scorer.pkl")

    # Metrics
    api.upload_file(
        path_or_fileobj=str(BASE / "output" / "metrics.json"),
        path_in_repo="metrics.json",
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        commit_message="Add metrics.json",
    )
    print("  ✓ metrics.json")

    # risk_scorer.py for reference
    api.upload_file(
        path_or_fileobj=str(BASE / "risk_scorer.py"),
        path_in_repo="risk_scorer.py",
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        commit_message="Add risk_scorer.py",
    )
    print("  ✓ risk_scorer.py")

    print(f"\n  Model live at: https://huggingface.co/{MODEL_REPO_ID}")


def deploy_space():
    print(f"\n{'─'*55}")
    print(f"  Deploying SPACE → {SPACE_REPO_ID}")
    print(f"{'─'*55}")

    create_repo(SPACE_REPO_ID, repo_type="space",
                space_sdk="streamlit", exist_ok=True, private=False)

    # Upload all files in hf_space/
    for f in SPACE_DIR.rglob("*"):
        if f.is_file():
            rel = f.relative_to(SPACE_DIR)
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=str(rel),
                repo_id=SPACE_REPO_ID,
                repo_type="space",
                commit_message=f"Add {rel}",
            )
            print(f"  ✓ {rel}")

    print(f"\n  Space live (building) at: https://huggingface.co/spaces/{SPACE_REPO_ID}")


if __name__ == "__main__":
    deploy_model()
    deploy_space()

    print(f"\n{'═'*55}")
    print("  Deployment complete!")
    print(f"  Model : https://huggingface.co/{MODEL_REPO_ID}")
    print(f"  Space : https://huggingface.co/spaces/{SPACE_REPO_ID}")
    print(f"{'═'*55}")
