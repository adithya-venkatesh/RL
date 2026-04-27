#!/usr/bin/env python3
"""
Diagnostic test to verify Pessimistic Q-Learning bug fixes.

This tests that after fixing the bugs, Pessimistic performs well in 
clean environments (~7.0 reward instead of -54.49).

Expected runtime: ~3-5 minutes
"""

import numpy as np
import sys
import os

# Import from your fixed code
try:
    # Adjust import based on your filename
    from main import (
        GridWorld, Q_Agent, play, play_att_with_belief, 
        play_pessimistic_clean
    )
except ImportError:
    print("ERROR: Could not import. Make sure main.py is in the same directory.")
    print("If your file has a different name, update the import above.")
    sys.exit(1)

def run_diagnostic():
    """Run diagnostic test on fixed Pessimistic implementation."""
    
    print("\n" + "="*70)
    print("DIAGNOSTIC TEST: PESSIMISTIC Q-LEARNING BUG FIXES")
    print("="*70)
    print("\nThis will verify that bug fixes are working correctly.")
    print("Expected clean environment reward: ~7.0")
    print("If you get -54.49 again, bugs are not fixed properly.")
    print("="*70)
    
    # Use fixed seed for reproducibility
    SEED = 42
    
    # ========== PHASE 1: Train Baseline ==========
    print("\n[Phase 1/3] Training baseline agent (2000 episodes)...")
    print("  This may take 1-2 minutes...")
    
    environment = GridWorld()
    environment.seed(SEED)
    agent = Q_Agent(environment)
    
    np.random.seed(SEED + 1000)
    reward_baseline = play(environment, agent, trials=2000, learn=True)
    baseline_avg = np.mean(reward_baseline[-100:])
    
    print(f"  ✓ Baseline training complete")
    print(f"    Final 100 episodes avg: {baseline_avg:.2f}")
    
    # ========== PHASE 2: Train Pessimistic ==========
    print("\n[Phase 2/3] Training Pessimistic Q-Learning (500 episodes)...")
    print("  This may take 1-2 minutes...")
    
    np.random.seed(SEED + 1000)
    reward_train = play_att_with_belief(environment, agent, trials=500, learn=True)
    train_avg = np.mean(reward_train)
    
    print(f"  ✓ Pessimistic training complete")
    print(f"    Training avg reward: {train_avg:.2f}")
    
    # ========== PHASE 3: Test Clean Environment ==========
    print("\n[Phase 3/3] Testing in CLEAN environment (100 episodes)...")
    print("  🔥 This is the critical test - should get ~7.0 if bugs are fixed!")
    
    np.random.seed(SEED + 1000)
    reward_clean, time_clean = play_pessimistic_clean(
        environment, agent, trials=100, measure_time=True
    )
    
    clean_avg = np.mean(reward_clean)
    clean_std = np.std(reward_clean)
    clean_min = np.min(reward_clean)
    clean_max = np.max(reward_clean)
    success_count = np.sum(np.array(reward_clean) > 5.0)
    
    # ========== RESULTS ==========
    print("\n" + "="*70)
    print("DIAGNOSTIC RESULTS")
    print("="*70)
    print(f"\nClean Environment Performance:")
    print(f"  Mean reward:     {clean_avg:7.2f} ± {clean_std:.2f}")
    print(f"  Min reward:      {clean_min:7.2f}")
    print(f"  Max reward:      {clean_max:7.2f}")
    print(f"  Success rate:    {success_count}/100 episodes")
    print(f"  Inference time:  {time_clean*1000:.4f} ms per action")
    
    # ========== VALIDATION ==========
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)
    
    checks_passed = 0
    total_checks = 3
    
    # Check 1: Mean reward
    print(f"\n[Check 1/3] Clean environment mean reward")
    if clean_avg > 5.0:
        print(f"  ✅ PASS: {clean_avg:.2f} > 5.0")
        checks_passed += 1
    else:
        print(f"  ❌ FAIL: {clean_avg:.2f} <= 5.0")
        print(f"    Expected: ~7.0, Got: {clean_avg:.2f}")
        print(f"    → Bugs may not be fixed properly!")
    
    # Check 2: Standard deviation
    print(f"\n[Check 2/3] Variance check")
    if clean_std < 10.0:
        print(f"  ✅ PASS: Std dev {clean_std:.2f} < 10.0")
        checks_passed += 1
    else:
        print(f"  ❌ FAIL: Std dev {clean_std:.2f} >= 10.0")
        print(f"    → High variance suggests instability")
    
    # Check 3: Success rate
    print(f"\n[Check 3/3] Success rate check")
    if success_count >= 70:
        print(f"  ✅ PASS: {success_count}/100 >= 70%")
        checks_passed += 1
    else:
        print(f"  ❌ FAIL: {success_count}/100 < 70%")
        print(f"    → Agent should succeed most of the time")
    
    # ========== FINAL VERDICT ==========
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    print(f"\nPassed {checks_passed}/{total_checks} validation checks")
    
    if checks_passed == total_checks:
        print("\n" + "🎉"*23)
        print("✅✅✅ SUCCESS! ALL BUGS FIXED! ✅✅✅")
        print("🎉"*23)
        print(f"\nPessimistic clean performance: {clean_avg:.2f}")
        print(f"This is comparable to Off-Policy's ~7.33")
        print("\n✓ You can now proceed to run full experiments!")
        print("✓ Both algorithms will be tested fairly in same conditions")
        print("\n" + "="*70)
        return True
        
    elif checks_passed >= 2:
        print("\n⚠️  PARTIAL SUCCESS")
        print(f"Most checks passed but {total_checks - checks_passed} failed.")
        print("Review the failed checks above.")
        print("\n" + "="*70)
        return False
        
    else:
        print("\n" + "❌"*23)
        print("❌❌❌ FAILURE - BUGS NOT FIXED ❌❌❌")
        print("❌"*23)
        print(f"\nPessimistic still getting: {clean_avg:.2f}")
        print(f"Expected: ~7.0")
        print("\nPossible issues:")
        print("  1. Bug fixes not applied correctly")
        print("  2. Wrong functions were modified")
        print("  3. Check get_att_state, get_policy_with_belief, get_policy")
        print("\n" + "="*70)
        return False

if __name__ == "__main__":
    print("\n" + "🔧"*35)
    print("PESSIMISTIC Q-LEARNING BUG FIX DIAGNOSTIC")
    print("🔧"*35)
    print("\nThis test will:")
    print("  1. Train a Pessimistic agent (same as experiments)")
    print("  2. Test in clean environment (no attacker)")
    print("  3. Verify performance is ~7.0 (not -54.49)")
    print("\nEstimated runtime: ~3-5 minutes")
    print("="*70)
    
    input("\nPress Enter to start diagnostic test...")
    
    success = run_diagnostic()
    
    if success:
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("\n1. Run 5 quick experiments:")
        print("   python main.py")
        print("   Choose option 1 (Quick test)")
        print("\n2. Verify both algorithms get ~7.0 in clean environment")
        print("\n3. If good, run full 50-101 experiments")
        print("\n4. Compare DQN attack performance fairly!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("TROUBLESHOOTING")
        print("="*70)
        print("\n1. Re-check Phase 1 fixes:")
        print("   - get_att_state: uses true_state not i")
        print("   - get_policy_with_belief: uses np.mean not min")
        print("   - get_policy: computes max-min policy")
        print("\n2. Make sure you edited the Q_Agent class")
        print("\n3. Restart Python/kernel after editing")
        print("\n4. Run this diagnostic again")
        print("="*70)