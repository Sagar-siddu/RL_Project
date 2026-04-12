#!/usr/bin/env python3
"""Install dependencies with compatible versions."""
import subprocess
import sys

packages = [
    "fastapi==0.110.0",
    "uvicorn[standard]==0.29.0",
    "pydantic==2.5.3",
    "openai>=1.0.0",
    "httpx",
    "openenv-core>=0.2.1",
]

for pkg in packages:
    print(f"\n{'='*60}")
    print(f"Installing: {pkg}")
    print('='*60)
    result = subprocess.run([sys.executable, "-m", "pip", "install", pkg])
    if result.returncode != 0:
        print(f"Warning: Failed to install {pkg}, continuing...")

print("\n" + "="*60)
print("Installation complete!")
print("="*60)
