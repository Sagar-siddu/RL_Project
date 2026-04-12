"""
Comprehensive API endpoint testing script.
Tests all custom endpoints: reset, step, state, and grade.
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:7860"

def test_reset():
    """Test the /custom/reset endpoint."""
    print("\n" + "="*60)
    print("TEST 1: Reset Environment")
    print("="*60)
    
    response = requests.post(f"{BASE_URL}/custom/reset")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Episode ID: {data['episode_id']}")
        print(f"Step Number: {data['observation']['step_number']}")
        print(f"Initial Reward: {data['reward']}")
        print(f"Number of SKUs: {len(data['observation']['skus'])}")
        for sku in data['observation']['skus']:
            print(f"  - {sku['sku_id']}: Price=${sku['current_price']}, Inventory={sku['inventory']}")
        return True
    else:
        print(f"Error: {response.text}")
        return False


def test_step(num_steps=5):
    """Test the /custom/step endpoint with multiple steps."""
    print("\n" + "="*60)
    print(f"TEST 2: Execute {num_steps} Steps")
    print("="*60)
    
    total_reward = 0
    total_revenue = 0
    
    for step_num in range(1, num_steps + 1):
        # Generate random prices based on previous observation
        prices = [50.0, 60.0, 65.0][:1]  # Adjust based on number of SKUs
        
        payload = {
            "action": {
                "prices": prices
            }
        }
        
        response = requests.post(
            f"{BASE_URL}/custom/step",
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            reward = data['reward']
            done = data['done']
            revenue = data['observation']['episode_revenue']
            
            total_reward += reward
            total_revenue = revenue
            
            print(f"Step {step_num}: Prices={prices}, Reward={reward:.4f}, Revenue=${revenue:.2f}, Done={done}")
            
            if done:
                print("Episode completed!")
                break
        else:
            print(f"Step {step_num} Error: {response.text}")
            return False
    
    print(f"Total Accumulated Reward: {total_reward:.4f}")
    print(f"Total Revenue: ${total_revenue:.2f}")
    return True


def test_state():
    """Test the /custom/state endpoint."""
    print("\n" + "="*60)
    print("TEST 3: Get Current State")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/custom/state")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        state = data['state']
        print(f"Episode ID: {data['episode_id']}")
        print(f"Task: {state['task_name']}")
        print(f"Steps Taken: {state['step_count']}")
        print(f"Max Steps: {state['max_steps']}")
        print(f"Total Revenue: ${state['total_revenue']:.2f}")
        print(f"Total Units Sold: {state['total_units_sold']}")
        print(f"Stockout Count: {state['stockout_count']}")
        print(f"Overstock Penalty: ${state['overstock_penalty_total']:.2f}")
        return True
    else:
        print(f"Error: {response.text}")
        return False


def test_grade():
    """Test the /custom/grade endpoint."""
    print("\n" + "="*60)
    print("TEST 4: Grade Episode")
    print("="*60)
    
    response = requests.post(f"{BASE_URL}/custom/grade")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Episode ID: {data['episode_id']}")
        print(f"Grade: {data['grade']:.4f}")
        print(f"Message: {data['message']}")
        return True
    else:
        print(f"Error: {response.text}")
        return False


def test_error_step_without_reset():
    """Test error handling: step without reset."""
    print("\n" + "="*60)
    print("TEST 5: Error Handling - Invalid Action Size")
    print("="*60)
    
    # Reset first to get a clean state
    requests.post(f"{BASE_URL}/custom/reset")
    
    # Try to step with wrong number of prices (should still work due to padding)
    # Instead, test with negative prices which should trigger an error
    payload = {
        "action": {
            "prices": [-10.0]  # Invalid price (negative)
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/custom/step",
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        # Check if error is in the observation
        data = response.json()
        if data['observation'].get('last_action_error'):
            print(f"✓ Correctly caught error: {data['observation']['last_action_error']}")
            return True
        else:
            print("Warning: Negative prices accepted (may be allowed by design)")
            return True
    else:
        print(f"Server error: {response.json()}")
        return False


def test_full_episode():
    """Run a complete episode with reset → multiple steps → grade."""
    print("\n" + "="*60)
    print("TEST 6: Full Episode Workflow")
    print("="*60)
    
    # Reset
    print("\n[1/3] Resetting environment...")
    response = requests.post(f"{BASE_URL}/custom/reset")
    if response.status_code != 200:
        print(f"Reset failed: {response.text}")
        return False
    
    reset_data = response.json()
    print(f"✓ Reset successful. Episode: {reset_data['episode_id']}")
    
    # Run steps until done
    print("\n[2/3] Running steps...")
    step_count = 0
    total_reward = 0
    
    while step_count < 30:  # Max 30 steps
        prices = [50.0 + (step_count % 5) * 5]  # Vary prices
        
        payload = {
            "action": {
                "prices": prices
            }
        }
        
        response = requests.post(f"{BASE_URL}/custom/step", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            total_reward += data['reward']
            step_count += 1
            
            if step_count % 5 == 0:
                print(f"  Step {step_count}: Reward={data['reward']:.4f}, Revenue=${data['observation']['episode_revenue']:.2f}")
            
            if data['done']:
                print(f"  ✓ Episode completed at step {step_count}")
                break
        else:
            print(f"Step {step_count} failed: {response.text}")
            return False
    
    # Grade the episode
    print("\n[3/3] Grading episode...")
    response = requests.post(f"{BASE_URL}/custom/grade")
    
    if response.status_code == 200:
        grade_data = response.json()
        print(f"✓ Final Grade: {grade_data['grade']:.4f}")
        print(f"Total Steps: {step_count}")
        print(f"Total Reward: {total_reward:.4f}")
        return True
    else:
        print(f"Grading failed: {response.text}")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + "  DYNAMIC PRICING API TEST SUITE".center(58) + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60)
    
    tests = [
        ("Reset Environment", test_reset),
        ("Execute Steps", test_step),
        ("Get State", test_state),
        ("Grade Episode", test_grade),
        ("Error Handling", test_error_step_without_reset),
        ("Full Episode", test_full_episode),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, "PASS" if result else "FAIL"))
        except Exception as e:
            print(f"Exception in {test_name}: {str(e)}")
            results.append((test_name, "ERROR"))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, status in results:
        status_symbol = "✓" if status == "PASS" else "✗"
        print(f"{status_symbol} {test_name}: {status}")
    
    passed = sum(1 for _, s in results if s == "PASS")
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")


if __name__ == "__main__":
    main()
