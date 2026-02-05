"""
Demonstrate Cascading Training - Run 3 short training sessions
This proves that loss continues from previous runs
"""

import os
import subprocess
import json
import time

def run_training_session(session_num, max_steps=10):
    """Run a single training session"""
    print(f"\n{'='*70}")
    print(f"TRAINING SESSION #{session_num}")
    print(f"{'='*70}\n")
    
    # Set environment variables for this run
    env = os.environ.copy()
    env['MAX_STEPS'] = str(max_steps)
    env['TOTAL_STEPS'] = '100000'
    
    # Run training
    result = subprocess.run(
        ['python', 'train.py'],
        env=env,
        capture_output=True,
        text=True
    )
    
    # Parse output for loss information
    output = result.stdout + result.stderr
    
    # Extract initial and final loss from output
    initial_loss = None
    final_loss = None
    
    for line in output.split('\n'):
        if 'INITIAL LOSS:' in line:
            try:
                initial_loss = float(line.split(':')[1].strip())
            except:
                pass
        if 'FINAL LOSS:' in line:
            try:
                final_loss = float(line.split(':')[1].strip())
            except:
                pass
        # Also capture from step logs
        if '[STEP' in line and 'Loss:' in line:
            try:
                parts = line.split('Loss:')[1].split(',')[0].strip()
                final_loss = float(parts)
            except:
                pass
    
    print(output)
    
    return {
        'session': session_num,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
    }

def main():
    print(f"\n{'='*70}")
    print("CASCADING TRAINING DEMONSTRATION")
    print(f"{'='*70}")
    print("\nThis will run 3 short training sessions to prove cascading works")
    print("Each session should start where the previous one ended\n")
    
    results = []
    
    # Run 3 training sessions
    for i in range(1, 4):
        result = run_training_session(i, max_steps=10)
        results.append(result)
        
        print(f"\n{'='*70}")
        print(f"SESSION #{i} COMPLETE")
        if result['initial_loss'] and result['final_loss']:
            print(f"Loss: {result['initial_loss']:.4f} → {result['final_loss']:.4f}")
        print(f"{'='*70}\n")
        
        # Small delay between runs
        time.sleep(2)
    
    # Analyze cascading
    print(f"\n{'='*70}")
    print("CASCADING ANALYSIS")
    print(f"{'='*70}\n")
    
    for i, result in enumerate(results):
        print(f"Session {result['session']}:")
        if result['initial_loss'] and result['final_loss']:
            print(f"  Initial: {result['initial_loss']:.4f}")
            print(f"  Final:   {result['final_loss']:.4f}")
            print(f"  Change:  {result['final_loss'] - result['initial_loss']:.4f}")
            
            if i > 0 and results[i-1]['final_loss']:
                prev_final = results[i-1]['final_loss']
                curr_initial = result['initial_loss']
                diff = abs(curr_initial - prev_final)
                
                if diff < 0.5:  # Allow small difference due to eval vs train mode
                    print(f"  ✓ CASCADING WORKS! (diff from prev: {diff:.4f})")
                else:
                    print(f"  ✗ CASCADING FAILED! (diff from prev: {diff:.4f})")
        else:
            print(f"  ⚠ Could not extract loss values")
        print()
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
