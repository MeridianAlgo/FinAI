#!/usr/bin/env python3
"""
FinAI Distributed Training Client
Submit tasks and monitor distributed training
"""
import sys
import json
import requests
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def submit_tasks_from_csv(server_url, csv_file="datasets.csv", auth_password="MeridianAlgo@TRAIN"):
    """Submit all pending datasets as tasks"""
    import csv
    
    if not Path(csv_file).exists():
        print(f"Error: {csv_file} not found")
        return
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        datasets = list(reader)
    
    # Load trained datasets
    trained_file = "trained_datasets.csv"
    trained_names = set()
    
    if Path(trained_file).exists():
        with open(trained_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            trained_names = {row['name'] for row in reader}
    
    # Submit pending datasets
    pending = [d for d in datasets if d['name'] not in trained_names]
    
    print(f"Found {len(pending)} pending datasets")
    print()
    
    submitted = 0
    for dataset in pending:
        try:
            response = requests.post(
                f"{server_url}/submit_task",
                json={
                    'dataset': dataset['name'],
                    'config': {
                        'config': dataset.get('config'),
                        'split': dataset.get('split', 'train')
                    },
                    'auth_password': auth_password
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"Submitted: {dataset['name']} (task_id: {data.get('task_id')})")
                submitted += 1
            else:
                print(f"Failed to submit {dataset['name']}: {response.text}")
        
        except Exception as e:
            print(f"Error submitting {dataset['name']}: {e}")
    
    print(f"\nSubmitted {submitted}/{len(pending)} tasks")

def show_status(server_url):
    """Show server status"""
    try:
        response = requests.get(f"{server_url}/status", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("="*80)
            print("Server Status")
            print("="*80)
            print(f"Status: {data['status']}")
            print(f"Workers: {data['workers']}")
            print(f"Pending tasks: {data['pending_tasks']}")
            print(f"Completed tasks: {data['completed_tasks']}")
            print()
        else:
            print(f"Error: {response.text}")
    
    except Exception as e:
        print(f"Error connecting to server: {e}")

def show_workers(server_url, auth_password="MeridianAlgo@TRAIN"):
    """Show all workers"""
    try:
        response = requests.get(
            f"{server_url}/workers",
            params={'auth_password': auth_password},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            workers = data['workers']
            
            print("="*80)
            print(f"Workers ({len(workers)})")
            print("="*80)
            
            if not workers:
                print("No workers connected")
            else:
                for worker_id, info in workers.items():
                    print(f"\n{worker_id}:")
                    print(f"  Status: {info['status']}")
                    print(f"  Current task: {info.get('current_task', {}).get('dataset', 'None')}")
                    stats = info.get('stats', {})
                    print(f"  Tasks completed: {stats.get('tasks_completed', 0)}")
                    print(f"  Total training time: {stats.get('total_training_time', 0):.2f}s")
            print()
        else:
            print(f"Error: {response.text}")
    
    except Exception as e:
        print(f"Error connecting to server: {e}")

def show_tasks(server_url, auth_password="MeridianAlgo@TRAIN"):
    """Show all tasks"""
    try:
        response = requests.get(
            f"{server_url}/tasks",
            params={'auth_password': auth_password},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            pending = data['pending']
            completed = data['completed']
            
            print("="*80)
            print(f"Tasks")
            print("="*80)
            
            print(f"\nPending ({len(pending)}):")
            if not pending:
                print("  None")
            else:
                for task in pending:
                    print(f"  {task['task_id']}: {task['dataset']}")
            
            print(f"\nCompleted ({len(completed)}):")
            if not completed:
                print("  None")
            else:
                for task in completed[-10:]:  # Show last 10
                    result = task.get('result', {})
                    status = result.get('status', 'unknown')
                    print(f"  {task['task_id']}: {status} by {task.get('worker_id', 'unknown')}")
            print()
        else:
            print(f"Error: {response.text}")
    
    except Exception as e:
        print(f"Error connecting to server: {e}")

def main():
    """Main client interface"""
    parser = argparse.ArgumentParser(description='FinAI Distributed Training Client')
    parser.add_argument('--server', required=True, help='Server URL (e.g., http://192.168.1.100:8765)')
    parser.add_argument('--password', default='MeridianAlgo@TRAIN', help='Authentication password')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Submit command
    submit_parser = subparsers.add_parser('submit', help='Submit tasks from datasets.csv')
    submit_parser.add_argument('--csv', default='datasets.csv', help='CSV file with datasets')
    
    # Status command
    subparsers.add_parser('status', help='Show server status')
    
    # Workers command
    subparsers.add_parser('workers', help='Show all workers')
    
    # Tasks command
    subparsers.add_parser('tasks', help='Show all tasks')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    server_url = args.server.rstrip('/')
    auth_password = args.password
    
    if args.command == 'submit':
        submit_tasks_from_csv(server_url, args.csv, auth_password)
    elif args.command == 'status':
        show_status(server_url)
    elif args.command == 'workers':
        show_workers(server_url, auth_password)
    elif args.command == 'tasks':
        show_tasks(server_url, auth_password)

if __name__ == "__main__":
    main()
