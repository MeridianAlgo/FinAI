import os
import subprocess
import time

env = os.environ.copy()
env["DTYPE"] = "float32"
env["SKIP_OPTIMIZER_SAVE"] = "1"
env["HARD_RAM_GUARD"] = "0"
env["MAX_RAM_GB"] = "40.0"
env["SOFT_RAM_GB"] = "30.0"
env["SOFT_RAM_PCT"] = "72"
env["MIN_THROTTLE_SEQ_LEN"] = "32"
env["DEBUG_STEPS"] = "0"
env["MAX_STEPS"] = "5"
env["GRAD_ACCUM"] = "8"
env["BATCH_SIZE"] = "1"
env["BLOCK_SIZE"] = "64"
env["USE_EWC"] = "1"
env["EWC_SAMPLES"] = "10"
env["GRADIENT_CHECKPOINTING"] = "0"
env["OPTIMIZER"] = "adafactor"
env["FREE_OPTIMIZER_BEFORE_FISHER"] = "1"
env["FISHER_SEQ_LEN"] = "32"
env["PYTHONUNBUFFERED"] = "1"
env["FORCE_BIN_SAVE"] = "1"

print("Starting training test for 5 steps...")
start = time.time()
result = subprocess.run(["python", "train.py"], env=env)
end = time.time()

elapsed = end - start
print("Finished!")
print(f"Stdout:\n{result.stdout}")
print(f"Stderr:\n{result.stderr}")
print(f"Total time: {elapsed:.2f} seconds")
