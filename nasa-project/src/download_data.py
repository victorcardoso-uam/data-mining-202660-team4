from pathlib import Path
from shutil import copytree

import kagglehub


DATASET = "darkmatternet/nasa-near-earth-asteroids-and-close-approaches"
RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    downloaded_path = Path(kagglehub.dataset_download(DATASET))
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    copytree(downloaded_path, RAW_DATA_DIR, dirs_exist_ok=True)
    print(f"Dataset files copied to: {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()
