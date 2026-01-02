"""Archive existing model/checkpoint artifacts and create fresh model/checkpoints directories.

This script moves `checkpoints/` and `downloaded_model/` into `legacy/backup-<ts>/` and
creates a new empty `checkpoints/` folder and `model/` placeholders for a new model.
"""
import os
import shutil
import time


def main():
    ts = time.strftime("%Y%m%d-%H%M%S")
    backup_dir = os.path.join("legacy", f"backup-{ts}")
    os.makedirs(backup_dir, exist_ok=True)

    for name in ("checkpoints", "downloaded_model"):
        if os.path.exists(name):
            dest = os.path.join(backup_dir, name)
            print(f"Moving {name} -> {dest}")
            shutil.move(name, dest)

    # recreate empty directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(os.path.join("checkpoints", "model"), exist_ok=True)
    os.makedirs("model", exist_ok=True)

    print(f"Archived old artifacts to {backup_dir} and created fresh `checkpoints/` and `model/`.")


if __name__ == "__main__":
    main()
