from comet_ml import API

api = API(api_key="dh60vNAugr3oTEfZeN9taJOr1")

workspace = "meridianalgo"
project_name = "meridian-ai"

experiments = api.get_experiments(workspace, project_name)
experiments.sort(key=lambda x: x.get_metadata()["startTimeMillis"], reverse=True)

print(f"Found {len(experiments)} experiments.")

for i, exp in enumerate(experiments[:3]):
    name = exp.get_name()
    metadata = exp.get_metadata()
    start_time = metadata.get("startTimeMillis", 0)
    end_time = metadata.get("endTimeMillis", 0)
    duration_min = (end_time - start_time) / 60000 if end_time else 0
    print(f"\nExperiment {i+1}: {name} (ID: {exp.id})")
    print(f"  Duration: {duration_min:.2f} min")

    # Get loss metrics
    metrics = exp.get_metrics("loss")
    if metrics:
        start_loss = float(metrics[0]["metricValue"])
        end_loss = float(metrics[-1]["metricValue"])
        print(f"  Start Loss: {start_loss:.4f}")
        print(f"  End Loss: {end_loss:.4f}")
        print(f"  Steps: {len(metrics)}")
    else:
        print("  No loss metrics found (might be still training).")
