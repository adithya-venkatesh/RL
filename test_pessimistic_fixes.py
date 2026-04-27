#!/usr/bin/env python3
"""
Diagnostic test to verify Pessimistic Q-Learning bug fixes.
"""

import numpy as np
import sys

# Import from your fixed code
try:
    from main import (
        GridWorld, Q_Agent, play, play_att_with_belief, 
        play_pessimistic_clean
    )
    print("✓ Successfully imported from main.py")
except ImportError as e:
    print(f"❌ ERROR: Could not import from main.py")
    print(f"Details: {e}")
    print("\nMake sure:")
    print("  1. File is named 'main.py' (not '2:23main.py' or 'main.py.txt')")
    print("  2. File is in the same directory as this test")
    sys.exit(1)

def run_diagnostic():
    """Run diagnostic test."""
    
    print("\n" + "="*70)
    print("DIAGNOSTIC TEST: PESSIMISTIC Q-LEARNING BUG FIXES")
    print("="*70)
    
    SEED = 42
    
    # Phase 1: Baseline
    print("\n[Phase 1/3] Training baseline (2000 episodes)...")
    environment = GridWorld()
    environment.seed(SEED)
    agent = Q_Agent(environment)
    
    np.random.seed(SEED + 1000)
    reward_baseline = play(environment, agent, trials=2000, learn=True)
    baseline_avg = np.mean(reward_baseline[-100:])
    print(f"  ✓ Baseline: {baseline_avg:.2f}")
    
    # Phase 2: Pessimistic Training
    print("\n[Phase 2/3] Training Pessimistic (500 episodes)...")
    np.random.seed(SEED + 1000)
    reward_train = play_att_with_belief(environment, agent, trials=500, learn=True)
    train_avg = np.mean(reward_train)
    print(f"  ✓ Training: {train_avg:.2f}")
    
    # Phase 3: Clean Test
    print("\n[Phase 3/3] Testing CLEAN environment (100 episodes)...")
    np.random.seed(SEED + 1000)
    reward_clean, time_clean = play_pessimistic_clean(
        environment, agent, trials=100, measure_time=True
    )
    
    clean_avg = np.mean(reward_clean)
    clean_std = np.std(reward_clean)
    success_count = np.sum(np.array(reward_clean) > 5.0)
    
    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Clean test: {clean_avg:.2f} ± {clean_std:.2f}")
    print(f"Success:    {success_count}/100")
    
    # Verdict
    if clean_avg > 5.0 and clean_std < 10.0:
        print("\n🎉 SUCCESS! Bugs are fixed!")
        return True
    else:
        print("\n❌ FAILURE - Bugs remain")
        print(f"Expected: ~7.0, Got: {clean_avg:.2f}")
        return False

if __name__ == "__main__":
    print("\n🔧 PESSIMISTIC Q-LEARNING BUG FIX DIAGNOSTIC 🔧\n")
    input("Press Enter to start...")
    
    success = run_diagnostic()
    
    if not success:
        print("\n📝 Next: Fix pareto_update function")
    
    print("\n" + "="*70)