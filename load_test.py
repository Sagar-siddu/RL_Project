"""
Load Testing Script for Dynamic Pricing Environment API
Tests concurrent episodes, high-frequency requests, and performance metrics.
"""

import asyncio
import requests
import time
import json
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, stdev
from typing import List, Dict, Tuple
import sys

BASE_URL = "http://127.0.0.1:7860"

class LoadTester:
    def __init__(self):
        self.results = {
            "create_episode": [],
            "step": [],
            "state": [],
            "grade": []
        }
        self.errors = []
        self.active_episodes = []
    
    def log(self, message: str, level: str = "INFO"):
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
    
    def create_episode(self, task_name: str = "single_sku_stable") -> Tuple[bool, str]:
        """Create a new episode and return success status and episode_id."""
        start = time.time()
        try:
            response = requests.post(
                f"{BASE_URL}/episode/create",
                params={"task_name": task_name},
                timeout=10
            )
            elapsed = time.time() - start
            self.results["create_episode"].append(elapsed)
            
            if response.status_code == 200:
                episode_id = response.json()["episode_id"]
                self.active_episodes.append(episode_id)
                return True, episode_id
            else:
                self.errors.append(f"Create failed: {response.text}")
                return False, None
        except Exception as e:
            self.errors.append(f"Create exception: {str(e)}")
            return False, None
    
    def execute_step(self, episode_id: str) -> bool:
        """Execute a step in an episode."""
        start = time.time()
        try:
            response = requests.post(
                f"{BASE_URL}/episode/{episode_id}/step",
                json={"action": {"prices": [50.0]}},
                timeout=10
            )
            elapsed = time.time() - start
            self.results["step"].append(elapsed)
            return response.status_code == 200
        except Exception as e:
            self.errors.append(f"Step exception: {str(e)}")
            return False
    
    def get_state(self, episode_id: str) -> bool:
        """Get state of an episode."""
        start = time.time()
        try:
            response = requests.get(
                f"{BASE_URL}/episode/{episode_id}/state",
                timeout=10
            )
            elapsed = time.time() - start
            self.results["state"].append(elapsed)
            return response.status_code == 200
        except Exception as e:
            self.errors.append(f"State exception: {str(e)}")
            return False
    
    def grade_episode(self, episode_id: str) -> bool:
        """Grade an episode."""
        start = time.time()
        try:
            response = requests.post(
                f"{BASE_URL}/episode/{episode_id}/grade",
                timeout=10
            )
            elapsed = time.time() - start
            self.results["grade"].append(elapsed)
            return response.status_code == 200
        except Exception as e:
            self.errors.append(f"Grade exception: {str(e)}")
            return False
    
    def cleanup(self):
        """Delete all active episodes."""
        for eid in self.active_episodes:
            try:
                requests.delete(f"{BASE_URL}/episode/{eid}", timeout=5)
            except:
                pass
    
    def print_stats(self):
        """Print performance statistics."""
        print("\n" + "="*70)
        print("LOAD TEST RESULTS".center(70))
        print("="*70)
        
        for operation, timings in self.results.items():
            if timings:
                print(f"\n{operation.upper()}:")
                print(f"  Count: {len(timings)}")
                print(f"  Min: {min(timings)*1000:.2f}ms")
                print(f"  Max: {max(timings)*1000:.2f}ms")
                print(f"  Avg: {mean(timings)*1000:.2f}ms")
                if len(timings) > 1:
                    print(f"  StdDev: {stdev(timings)*1000:.2f}ms")
        
        if self.errors:
            print(f"\nERRORS: {len(self.errors)}")
            for err in self.errors[:5]:
                print(f"  - {err}")
            if len(self.errors) > 5:
                print(f"  ... and {len(self.errors) - 5} more")
        
        print(f"\nActive Episodes: {len(self.active_episodes)}")
        print("="*70)


def test_concurrent_episodes(num_episodes: int = 5):
    """Test concurrent episode creation and execution."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "TEST 1: CONCURRENT EPISODES".center(68) + "█")
    print("█" + f"Creating and running {num_episodes} episodes in parallel...".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    tester = LoadTester()
    
    def run_episode(idx):
        tester.log(f"Episode {idx}: Creating...")
        success, episode_id = tester.create_episode()
        if not success:
            return
        
        tester.log(f"Episode {idx}: Running steps...")
        for step in range(5):
            if not tester.execute_step(episode_id):
                tester.log(f"Episode {idx}: Step failed", "WARN")
                break
        
        tester.log(f"Episode {idx}: Grading...")
        tester.grade_episode(episode_id)
    
    # Run episodes concurrently
    with ThreadPoolExecutor(max_workers=num_episodes) as executor:
        futures = [executor.submit(run_episode, i) for i in range(num_episodes)]
        for future in futures:
            future.result()
    
    tester.print_stats()
    tester.cleanup()


def test_high_frequency_requests(num_requests: int = 100):
    """Test high frequency of requests on single episode."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "TEST 2: HIGH FREQUENCY REQUESTS".center(68) + "█")
    print("█" + f"Sending {num_requests} rapid requests to single episode...".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    tester = LoadTester()
    
    # Create single episode
    tester.log("Creating episode...")
    success, episode_id = tester.create_episode()
    if not success:
        tester.log("Failed to create episode", "ERROR")
        return
    
    tester.log(f"Sending {num_requests} rapid step requests...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(tester.execute_step, episode_id)
            for _ in range(num_requests)
        ]
        completed = sum(1 for f in futures if f.result())
    
    tester.log(f"Completed {completed}/{num_requests} requests")
    tester.print_stats()
    tester.cleanup()


def test_mixed_workload(duration_seconds: int = 30):
    """Test mixed workload with varied operations."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "TEST 3: MIXED WORKLOAD".center(68) + "█")
    print("█" + f"Running mixed operations for {duration_seconds}s...".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    tester = LoadTester()
    start_time = time.time()
    episode_count = 0
    operation_count = 0
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        
        while time.time() - start_time < duration_seconds:
            # Create new episodes
            future = executor.submit(tester.create_episode)
            futures.append(("create", future))
            episode_count += 1
            
            # Execute steps on existing episodes
            for eid in tester.active_episodes[-3:]:  # On recent episodes
                future = executor.submit(tester.execute_step, eid)
                futures.append(("step", future))
                operation_count += 1
            
            # Get state
            if tester.active_episodes:
                future = executor.submit(tester.get_state, tester.active_episodes[-1])
                futures.append(("state", future))
                operation_count += 1
            
            time.sleep(0.5)
    
    tester.log(f"Created {episode_count} episodes, {operation_count} operations")
    tester.print_stats()
    tester.cleanup()


def test_stress_limits():
    """Test server limits - create many episodes."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "TEST 4: STRESS TEST - EPISODE LIMITS".center(68) + "█")
    print("█" + "How many concurrent episodes can the server handle?".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    tester = LoadTester()
    max_episodes = 50
    
    for i in range(max_episodes):
        success, eid = tester.create_episode()
        if not success:
            tester.log(f"Server limit reached at {i} episodes", "WARN")
            break
        if (i + 1) % 10 == 0:
            tester.log(f"Created {i + 1} episodes")
    
    response = requests.get(f"{BASE_URL}/episodes/list")
    active = response.json()["total_episodes"]
    tester.log(f"Total active episodes: {active}")
    
    # Try operations
    if tester.active_episodes:
        tester.log("Testing operations on stress-loaded server...")
        for eid in tester.active_episodes[:5]:
            tester.execute_step(eid)
            tester.get_state(eid)
    
    print(f"\nSuccessfully created {len(tester.active_episodes)} concurrent episodes")
    tester.cleanup()


def test_different_tasks():
    """Test creating episodes with different task types."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "TEST 5: DIFFERENT TASK TYPES".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    tasks = ["single_sku_stable", "multi_sku_competitors", "demand_shocks_perishables"]
    tester = LoadTester()
    
    for task in tasks:
        tester.log(f"Testing task: {task}")
        success, eid = tester.create_episode(task_name=task)
        
        if success:
            # Run a few steps
            for _ in range(3):
                tester.execute_step(eid)
            
            # Get state
            tester.get_state(eid)
            
            # Grade
            tester.grade_episode(eid)
            tester.log(f"  ✓ {task} completed successfully")
        else:
            tester.log(f"  ✗ {task} failed", "ERROR")
    
    tester.print_stats()
    tester.cleanup()


def main():
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "DYNAMIC PRICING - LOAD TEST SUITE".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    tests = [
        ("Concurrent Episodes", lambda: test_concurrent_episodes(5)),
        ("High Frequency Requests", lambda: test_high_frequency_requests(50)),
        ("Mixed Workload", lambda: test_mixed_workload(20)),
        ("Stress Test - Episode Limits", lambda: test_stress_limits()),
        ("Different Task Types", lambda: test_different_tasks()),
    ]
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"\n✓ {test_name} completed\n")
        except Exception as e:
            print(f"\n✗ {test_name} failed: {str(e)}\n")
            import traceback
            traceback.print_exc()
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "ALL LOAD TESTS COMPLETED".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")


if __name__ == "__main__":
    main()
