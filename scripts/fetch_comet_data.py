import os
import json
from comet_ml import API
from dotenv import load_dotenv

load_dotenv()


def fetch_data():
    api_key = os.getenv("COMET_API_KEY")
    if not api_key:
        print("COMET_API_KEY not found in .env")
        return

    api = API(api_key=api_key)
    workspace = "meridianalgo"
    project_name = "finai-next"

    print(f"Fetching experiments from {workspace}/{project_name}...")
    experiments = api.get_experiments(workspace, project_name)

    # Sort by creation time, descending
    experiments.sort(key=lambda x: x.start_server_timestamp, reverse=True)

    last_runs = experiments[:10]  # Get last 10 just to be safe

    data = []
    for exp in last_runs:
        exp_data = {
            "name": exp.name,
            "id": exp.id,
            "start_time": exp.start_server_timestamp,
            "metrics": [],
        }

        print(f"Fetching metrics for experiment: {exp.name or exp.id}")
        # Fetch loss metrics
        metrics = exp.get_metrics("loss")
        for m in metrics:
            exp_data["metrics"].append(
                {
                    "step": m.get("step"),
                    "value": m.get("metricValue"),
                    "timestamp": m.get("timestamp"),
                }
            )

        # Also log parameters to see if it's resuming
        params = exp.get_parameters_summary()
        exp_data["parameters"] = {p["name"]: p["valueMax"] for p in params}

        data.append(exp_data)

    with open("comet_logs.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved data for {len(data)} runs to comet_logs.json")


if __name__ == "__main__":
    fetch_data()
