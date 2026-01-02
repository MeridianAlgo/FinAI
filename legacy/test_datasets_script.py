"""Legacy dataset test script moved here to avoid accidental execution during pytest.

To run manually:
    python legacy/test_datasets_script.py
"""

from tests.test_datasets import *  # noqa: F401,F403

if __name__ == "__main__":
    print("Run dataset checks manually using legacy/test_datasets_script.py")
