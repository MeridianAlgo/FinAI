import json


def summarize_logs():
    try:
        with open("comet_logs.json", "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading comet_logs.json: {e}")
        return

    summary = []
    summary.append("Training Run Loss Summary")
    summary.append("========================\n")

    # Experiments were fetched in reverse chronological order (newest first)
    # Let's reverse to show them in order of occurrence if possible, or just keep newest first.
    # Actually, chronological order might be better to see the "path".
    data.sort(key=lambda x: x.get("start_time", 0))

    for run in data:
        metrics = run.get("metrics", [])
        if not metrics:
            continue

        # Get start and end loss
        start_loss = float(metrics[0]["value"])
        end_loss = float(metrics[-1]["value"])
        start_step = metrics[0]["step"]
        end_step = metrics[-1]["step"]

        summary.append(f"Run: {run['name'] or run['id']}")
        summary.append(f"  Start: Step {start_step}, Loss {start_loss:.4f}")
        summary.append(f"  End:   Step {end_step}, Loss {end_loss:.4f}")

        if start_loss > 15:
            summary.append("  [!] ISSUE: Loss started > 15. Weights were likely reset.")

        summary.append("")

    with open("comet_summary.txt", "w") as f:
        f.write("\n".join(summary))

    print("Created comet_summary.txt")


if __name__ == "__main__":
    summarize_logs()
