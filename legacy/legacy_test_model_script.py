"""Legacy manual test harness moved here to avoid pytest side-effects.

Run manually for interactive debugging: `python legacy/legacy_test_model_script.py`
"""

from tests._legacy_test_model import *  # noqa: F401,F403

if __name__ == "__main__":
    print("Run legacy model checks from tests/_legacy_test_model.py manually")
