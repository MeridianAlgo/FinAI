import os
import tempfile

import pytest
from fin_ai.model.transformer import FinAIModel
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

HF_REPO = os.environ.get("HF_REPO_ID", "MeridianAlgo/Fin.AI")


def _get_repo_files_info(repo_id):
    api = HfApi()
    try:
        info = api.model_info(repo_id)
    except Exception as e:
        pytest.skip(f"Unable to query HF repo {repo_id}: {e}")
    # siblings contain file metadata including rfilename and size
    files = {s.rfilename: getattr(s, "size", None) for s in info.siblings}
    return files


@pytest.mark.slow
def test_hf_repo_has_model_and_config():
    files = _get_repo_files_info(HF_REPO)
    assert "config.json" in files, "config.json missing from HF repo"

    # Check for common model file names
    model_candidates = ["model.pt", "pytorch_model.bin", "pytorch_model.safetensors"]
    found = [f for f in model_candidates if f in files]
    assert (
        found
    ), f"No model weight files found in {HF_REPO}. Found files: {list(files.keys())[:20]}"


@pytest.mark.slow
def test_tokenizer_and_config_download():
    # Download tokenizer and config locally (these are small files)
    with tempfile.TemporaryDirectory() as tmp:
        tokenizer = AutoTokenizer.from_pretrained(HF_REPO, cache_dir=tmp)
        # tokenizer loaded successfully
        assert hasattr(tokenizer, "encode")


@pytest.mark.slow
def test_download_small_model_and_load_if_small():
    files = _get_repo_files_info(HF_REPO)
    # Prefer safetensors or model.pt if available
    preferred = None
    for name in ["pytorch_model.safetensors", "model.pt", "pytorch_model.bin"]:
        if name in files:
            preferred = (name, files[name])
            break

    if not preferred:
        pytest.skip("No model weight files available to download in the repo")

    name, size = preferred
    # If size is unknown or very large, skip to avoid downloading huge models
    if size is None or size > 200 * 1024 * 1024:
        pytest.skip(
            f"Model file {name} is too large ({size}) to download in CI/test environment"
        )

    with tempfile.TemporaryDirectory() as tmp:
        # Download the model file into tmp dir
        hf_hub_download(repo_id=HF_REPO, filename=name, local_dir=tmp)
        # Also ensure config.json presence
        hf_hub_download(repo_id=HF_REPO, filename="config.json", local_dir=tmp)

        # Attempt to load model using local path
        model = FinAIModel.from_pretrained(tmp)
        # Run a short forward pass
        import torch

        model.eval()
        input_ids = torch.randint(0, model.config.vocab_size, (1, 8))
        out = model(input_ids)
        assert "logits" in out
