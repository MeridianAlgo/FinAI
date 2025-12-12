#!/usr/bin/env python3
import json
import os
import csv
import time
import random
import subprocess
from datetime import datetime
from datasets import load_dataset

STEPS_PER_RUN = 5000
MAX_TRAINING_TIME = 8400

TEST_QUESTIONS = [
    "What is a stock?", "How do I start investing?", "What is compound interest?",
    "Explain diversification", "What is a 401k?", "How to save for retirement?",
    "What is inflation?", "Explain bonds vs stocks", "What is market capitalization?",
    "How to read a balance sheet?", "What is ROI?", "Explain ETFs",
    "What is a dividend?", "How to calculate profit margin?", "What is liquidity?",
    "What is a bear market?", "Explain P/E ratio", "What is dollar cost averaging?",
    "How do options work?", "What is a hedge fund?"
]

def load_datasets_config():
    datasets = []
    if os.path.exists("datasets.csv"):
        with open("datasets.csv", 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('enabled', 'true').lower() == 'true':
                    datasets.append({
                        "name": row['name'],
                        "split": row.get('split', 'train'),
                        "question_field": row['question_field'],
                        "answer_field": row['answer_field']
                    })
    return datasets

def load_trained_state():
    state = {}
    if os.path.exists("trained_datasets.csv"):
        with open("trained_datasets.csv", 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                state[row['dataset']] = {
                    'total_samples': int(row.get('total_samples', 0)),
                    'steps_completed': int(row.get('steps_completed', 0)),
                    'steps_remaining': int(row.get('steps_remaining', 0)),
                    'last_trained': row.get('last_trained', ''),
                    'status': row.get('status', 'pending'),
                    'error': row.get('error', ''),
                    'success_count': int(row.get('success_count', 0)),
                    'fail_count': int(row.get('fail_count', 0))
                }
    return state

def save_trained_state(state):
    with open("trained_datasets.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'dataset', 'total_samples', 'steps_completed', 'steps_remaining',
            'last_trained', 'status', 'error', 'success_count', 'fail_count'
        ])
        writer.writeheader()
        for name, data in state.items():
            writer.writerow({
                'dataset': name,
                'total_samples': data.get('total_samples', 0),
                'steps_completed': data.get('steps_completed', 0),
                'steps_remaining': data.get('steps_remaining', 0),
                'last_trained': data.get('last_trained', ''),
                'status': data.get('status', 'pending'),
                'error': data.get('error', ''),
                'success_count': data.get('success_count', 0),
                'fail_count': data.get('fail_count', 0)
            })

class TrainingState:
    def __init__(self):
        self.state_file = "training_state.json"
        self.state = self.load_state()
        self.datasets = load_datasets_config()
        self.trained = load_trained_state()
    
    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "current_dataset_idx": 0,
            "total_steps": 0,
            "cycle_count": 0,
            "version": "1.0.0",
            "releases_created": 0,
            "last_run": None,
            "total_success": 0,
            "total_failures": 0
        }
    
    def save_state(self):
        self.state["last_run"] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        save_trained_state(self.trained)
    
    def get_current_dataset(self):
        if not self.datasets:
            return None
        return self.datasets[self.state["current_dataset_idx"] % len(self.datasets)]
    
    def get_dataset_progress(self, name):
        return self.trained.get(name)
    
    def init_dataset(self, name, total_samples):
        if name not in self.trained:
            self.trained[name] = {
                'total_samples': total_samples,
                'steps_completed': 0,
                'steps_remaining': total_samples,
                'last_trained': '',
                'status': 'pending',
                'error': '',
                'success_count': 0,
                'fail_count': 0
            }
        elif self.trained[name]['total_samples'] == 0:
            self.trained[name]['total_samples'] = total_samples
            self.trained[name]['steps_remaining'] = total_samples - self.trained[name]['steps_completed']
    
    def record_success(self, name, steps_trained):
        if name in self.trained:
            self.trained[name]['steps_completed'] += steps_trained
            self.trained[name]['steps_remaining'] = max(0, self.trained[name]['total_samples'] - self.trained[name]['steps_completed'])
            self.trained[name]['last_trained'] = datetime.now().isoformat()
            self.trained[name]['success_count'] += 1
            self.trained[name]['error'] = ''
            
            if self.trained[name]['steps_remaining'] == 0:
                self.trained[name]['status'] = 'completed'
            else:
                self.trained[name]['status'] = 'in_progress'
        
        self.state['total_steps'] += steps_trained
        self.state['total_success'] = self.state.get('total_success', 0) + 1
    
    def record_failure(self, name, error_msg):
        if name not in self.trained:
            self.trained[name] = {
                'total_samples': 0,
                'steps_completed': 0,
                'steps_remaining': 0,
                'last_trained': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(error_msg)[:200],
                'success_count': 0,
                'fail_count': 1
            }
        else:
            self.trained[name]['fail_count'] += 1
            self.trained[name]['error'] = str(error_msg)[:200]
            self.trained[name]['status'] = 'failed'
            self.trained[name]['last_trained'] = datetime.now().isoformat()
        
        self.state['total_failures'] = self.state.get('total_failures', 0) + 1
    
    def should_recycle_dataset(self, name):
        """Check if dataset should be recycled (completed all samples)"""
        if name in self.trained:
            return self.trained[name]['steps_remaining'] <= 0 and self.trained[name]['total_samples'] > 0
        return False
    
    def recycle_dataset(self, name):
        """Reset dataset for retraining"""
        if name in self.trained:
            self.trained[name]['steps_completed'] = 0
            self.trained[name]['steps_remaining'] = self.trained[name]['total_samples']
            self.trained[name]['status'] = 'recycled'
            self.trained[name]['error'] = ''
            print(f"♻️ Recycled {name} for retraining")
    
    def advance_dataset(self):
        self.state["current_dataset_idx"] = (self.state["current_dataset_idx"] + 1) % len(self.datasets)
        if self.state["current_dataset_idx"] == 0:
            self.state["cycle_count"] += 1
            return True
        return False
    
    def increment_version(self):
        parts = self.state["version"].split(".")
        parts[2] = str(int(parts[2]) + 1)
        self.state["version"] = ".".join(parts)
        self.state["releases_created"] += 1
        return self.state["version"]
    
    def check_and_recycle_all(self):
        """Check if all datasets are complete and recycle them"""
        all_complete = True
        for ds in self.datasets:
            name = ds['name']
            if name in self.trained:
                if self.trained[name]['steps_remaining'] > 0 or self.trained[name]['total_samples'] == 0:
                    all_complete = False
                    break
            else:
                all_complete = False
                break
        
        if all_complete:
            print("\n♻️ All datasets complete! Recycling for next round...")
            for ds in self.datasets:
                self.recycle_dataset(ds['name'])
            return True
        return False

def get_dataset_size(dataset_config):
    try:
        ds = load_dataset(dataset_config['name'])
        split = dataset_config['split']
        if split not in ds:
            split = list(ds.keys())[0]
        return len(ds[split]), None
    except Exception as e:
        return 0, str(e)

def prepare_dataset(dataset_config, start_idx=0, count=5000):
    print(f"\n📥 Loading {dataset_config['name']} (idx {start_idx}, count {count})...")
    try:
        ds = load_dataset(dataset_config['name'])
        split = dataset_config['split']
        if split not in ds:
            split = list(ds.keys())[0]
        
        split_data = ds[split]
        total = len(split_data)
        print(f"   Total: {total}")
        
        training_data = []
        q_field = dataset_config['question_field']
        a_field = dataset_config['answer_field']
        end_idx = min(start_idx + count, total)
        
        for i in range(start_idx, end_idx):
            try:
                item = split_data[i]
                q = str(item.get(q_field, '')).strip() if item.get(q_field) else None
                a = str(item.get(a_field, '')).strip() if item.get(a_field) else None
                
                if q_field == a_field and q:
                    a = q
                    q = "Explain: " + q[:100]
                
                if q and a and len(q) > 5 and len(a) > 5:
                    training_data.append({'question': q.lower(), 'answer': a.lower()})
            except:
                continue
        
        if len(training_data) < 10:
            return [], total, "Not enough valid examples"
        
        print(f"   ✅ Prepared {len(training_data)} examples")
        return training_data, total, None
    except Exception as e:
        return [], 0, str(e)

def train_on_data(training_data, steps):
    if not training_data:
        return 0, "No training data"
    
    temp_file = "temp_training.txt"
    with open(temp_file, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(f"user: {item['question']}\nassistant: {item['answer']}\n\n")
    
    try:
        result = os.system(f"python main.py train {temp_file} --steps {steps}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        if result != 0:
            return 0, f"Training failed with code {result}"
        return steps, None
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return 0, str(e)

def test_model():
    print("\n🧪 Testing model...")
    results = []
    questions = random.sample(TEST_QUESTIONS, 10)
    for q in questions:
        try:
            result = subprocess.run(["python", "main.py", "ask", q], capture_output=True, text=True, timeout=60)
            answer = result.stdout.strip()[:500]
            results.append({"question": q, "answer": answer if answer else "No response"})
        except Exception as e:
            results.append({"question": q, "answer": f"Error: {str(e)[:100]}"})
    return results

def create_release(state, test_results):
    version = state.state["version"]
    tag = f"v{version}"
    
    qa_section = ""
    for i, qa in enumerate(test_results, 1):
        qa_section += f"### Q{i}: {qa['question']}\n> {qa['answer'][:300]}\n\n"
    
    datasets_section = ""
    for name, data in state.trained.items():
        short = name.split("/")[-1]
        pct = (data['steps_completed'] / data['total_samples'] * 100) if data['total_samples'] > 0 else 0
        status_icon = "✅" if data['status'] == 'completed' else "🔄" if data['status'] == 'in_progress' else "❌" if data['status'] == 'failed' else "⏳"
        datasets_section += f"| {short} | {data['steps_completed']:,}/{data['total_samples']:,} | {pct:.1f}% | {status_icon} | {data['success_count']}/{data['fail_count']} |\n"
    
    notes = f"""# 🤖 FinAI {tag}

## 📊 Stats
| Metric | Value |
|:------:|:-----:|
| Total Steps | **{state.state['total_steps']:,}** |
| Cycles | **{state.state['cycle_count']}** |
| Datasets | **{len(state.trained)}** |
| Success/Fail | **{state.state.get('total_success', 0)}/{state.state.get('total_failures', 0)}** |

## 📈 Dataset Progress
| Dataset | Progress | % | Status | S/F |
|:-------:|:--------:|:-:|:------:|:---:|
{datasets_section}

## 🧪 Sample Responses
{qa_section}

---
*Auto-generated by FinAI*
"""
    
    with open("RELEASE_NOTES.md", "w") as f:
        f.write(notes)
    
    print(f"\n📦 Creating release {tag}...")
    os.system(f'gh release delete {tag} -y 2>/dev/null || true')
    os.system(f'git tag -d {tag} 2>/dev/null || true')
    os.system(f'git push origin :refs/tags/{tag} 2>/dev/null || true')
    os.system(f'git tag {tag}')
    os.system(f'git push origin {tag}')
    
    cmd = f'gh release create {tag} --title "FinAI {tag}" --notes-file RELEASE_NOTES.md'
    if os.path.exists("models/finai_gpt.pt"):
        cmd += " models/finai_gpt.pt"
    cmd += " training_state.json trained_datasets.csv"
    os.system(cmd)
    
    if os.path.exists("RELEASE_NOTES.md"):
        os.remove("RELEASE_NOTES.md")
    print(f"✅ Release {tag} created!")

def main():
    print("=" * 60)
    print("🤖 FinAI Training")
    print("=" * 60)
    
    state = TrainingState()
    
    if not state.datasets:
        print("❌ No datasets in datasets.csv!")
        return
    
    print(f"\n📊 v{state.state['version']} | {state.state['total_steps']:,} steps | Cycle {state.state['cycle_count']}")
    print(f"📚 {len(state.datasets)} datasets | ✅ {state.state.get('total_success', 0)} | ❌ {state.state.get('total_failures', 0)}")
    
    # Check if we need to recycle
    state.check_and_recycle_all()
    
    start_time = time.time()
    consecutive_failures = 0
    
    while True:
        if time.time() - start_time > MAX_TRAINING_TIME:
            print(f"\n⏰ Time limit")
            break
        
        if consecutive_failures >= len(state.datasets):
            print(f"\n❌ All datasets failed this round")
            break
        
        dataset = state.get_current_dataset()
        if not dataset:
            break
        
        name = dataset['name']
        progress = state.get_dataset_progress(name)
        
        # Check if needs recycling
        if progress and state.should_recycle_dataset(name):
            state.recycle_dataset(name)
            progress = state.get_dataset_progress(name)
        
        # Initialize if first time
        if not progress or progress['total_samples'] == 0:
            print(f"\n🔍 Discovering {name}...")
            total, error = get_dataset_size(dataset)
            if total == 0:
                print(f"   ❌ Failed: {error}")
                state.record_failure(name, error or "Could not load dataset")
                consecutive_failures += 1
                state.advance_dataset()
                state.save_state()
                continue
            state.init_dataset(name, total)
            progress = state.get_dataset_progress(name)
        
        # Skip if no remaining
        if progress['steps_remaining'] <= 0:
            state.advance_dataset()
            state.save_state()
            continue
        
        # Train
        start_idx = progress['steps_completed']
        steps_to_train = min(STEPS_PER_RUN, progress['steps_remaining'])
        
        print(f"\n🎯 {name}: {start_idx:,}/{progress['total_samples']:,} (+{steps_to_train})")
        
        training_data, total, error = prepare_dataset(dataset, start_idx, steps_to_train)
        
        if error or not training_data:
            print(f"   ❌ {error or 'No data'}")
            state.record_failure(name, error or "No training data")
            consecutive_failures += 1
            state.advance_dataset()
            state.save_state()
            continue
        
        steps_trained, train_error = train_on_data(training_data, len(training_data))
        
        if train_error or steps_trained == 0:
            print(f"   ❌ {train_error or 'Training failed'}")
            state.record_failure(name, train_error or "Training failed")
            consecutive_failures += 1
            state.advance_dataset()
            state.save_state()
            continue
        
        # Success!
        print(f"   ✅ Trained {steps_trained} steps")
        consecutive_failures = 0
        state.record_success(name, steps_trained)
        
        cycle_done = state.advance_dataset()
        
        if cycle_done:
            print(f"\n🔄 Cycle {state.state['cycle_count']} complete!")
            state.check_and_recycle_all()
            
            if state.state['cycle_count'] % 3 == 0:
                state.increment_version()
                results = test_model()
                create_release(state, results)
        
        state.save_state()
        
        if MAX_TRAINING_TIME - (time.time() - start_time) < 600:
            break
    
    state.save_state()
    print(f"\n✅ Done | {state.state['total_steps']:,} steps | ✅{state.state.get('total_success',0)} ❌{state.state.get('total_failures',0)}")

if __name__ == "__main__":
    main()
