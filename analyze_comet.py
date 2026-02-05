import json

with open('comet_logs.json') as f:
    data = json.load(f)

print(f'Total runs: {len(data)}\n')

for i, run in enumerate(data[:6]):
    name = run['name']
    if run['metrics']:
        start_step = run['metrics'][0]['step']
        start_loss = float(run['metrics'][0]['value'])
        end_step = run['metrics'][-1]['step']
        end_loss = float(run['metrics'][-1]['value'])
        print(f"Run {i+1}: {name}")
        print(f"  Start: Step {start_step}, Loss {start_loss:.4f}")
        print(f"  End:   Step {end_step}, Loss {end_loss:.4f}")
        print()
    else:
        print(f"Run {i+1}: {name} - No metrics")
        print()
