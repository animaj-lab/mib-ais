from pathlib import Path

from huggingface_hub import snapshot_download


def ensure_hf_dataset(download_root: Path) -> None:
    """Download the HuggingFace dataset into *download_root* if not already present.

    The dataset contains two subdirectories: ``in_house_dataset/`` and``prod_test_dataset/``.  The check uses the
    presence of ``in_house_dataset/`` as a proxy for a complete download.

    Args:
        download_root: Local directory that will contain the two dataset subdirectories after download.
    """

    if (download_root / "in_house_dataset").exists():
        return

    download_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="AnimajSAS/mib_rig_controllers_values",
        repo_type="dataset",
        local_dir=download_root,
    )
