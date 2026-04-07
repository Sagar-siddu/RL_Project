#!/usr/bin/env python3
"""Quick API test - simplified version."""
import httpx
import time

API_URL = "http://127.0.0.1:7860"
client = httpx.Client(timeout=60.0)

try:
    print("=" * 60)
    print("QUICK API TEST")
    print("=" * 60)
    
    # Test 1: Reset
    print("\n[1/4] Testing /reset endpoint...")
    resp = client.post(f"{API_URL}/reset")
    if resp.status_code == 200:
        data = resp.json()
        obs = data.get("observation", {})
        print(f"✓ SUCCESS - Task: {obs.get('task_name')}")
    else:
        print(f"✗ FAILED - Status: {resp.status_code}")
        print(resp.text)
        exit(1)
    
    # Test 2: Step
    print("\n[2/4] Testing /step endpoint...")
    step_resp = client.post(f"{API_URL}/step", json={"action": {"value": [49.99]}})
    if step_resp.status_code == 200:
        step_data = step_resp.json()
        print(f"✓ SUCCESS - Reward: {step_data.get('reward'):.4f}")
    else:
        print(f"✗ FAILED - Status: {step_resp.status_code}")
        print(step_resp.text)
        exit(1)
    
    # Test 3: Multiple steps
    print("\n[3/4] Running 5 more steps...")
    for i in range(5):
        resp = client.post(f"{API_URL}/step", json={"action": {"value": [45.0 + i * 2]}})
        if resp.status_code == 200:
            data = resp.json()
            reward = data.get('reward', 0.0)
            print(f"  Step {i+2}: Reward={reward:.4f}")
        else:
            print(f"  Step {i+2}: FAILED")
    
    # Test 4: Grade
    print("\n[4/4] Testing /grade endpoint...")
    grade_resp = client.post(f"{API_URL}/grade")
    if grade_resp.status_code == 200:
        grade_data = grade_resp.json()
        grade = grade_data.get('grade', 0.0)
        print(f"✓ SUCCESS - Episode Grade: {grade:.4f}")
    else:
        print(f"✗ FAILED - Status: {grade_resp.status_code}")
        print(grade_resp.text)
        exit(1)
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    client.close()
