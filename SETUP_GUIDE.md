# Environment Setup Guide - Python 3.15 Alpha Issue & Solutions

## Current Issue

Your workspace is using **Python 3.15 alpha**, which lacks pre-compiled wheels for packages like `pydantic-core`. This causes build failures when trying to install packages that require Rust/Cargo compilation.

```
Error: × Encountered error while generating package metadata.
╰─> pydantic-core

note: This package requires Rust and Cargo to compile extensions.
```

---

## Solutions

### ✅ Solution 1: Use Python 3.11 (Recommended)

**Pros**: Pre-compiled wheels available, stable, widely supported
**Time**: 5 minutes

#### Steps:

1. **Install Python 3.11** (if not already)
   - Windows: Download from python.org or use Windows Store
   - macOS: `brew install python@3.11`
   - Linux: `apt-get install python3.11`

2. **Create new virtual environment with Python 3.11**
   ```bash
   cd c:\Users\PETERSUNNY\OneDrive\Documents\GitHub\RL_Project\dynamic_pricing_env
   python3.11 -m venv venv_py311
   source venv_py311/bin/activate  # On Windows: venv_py311\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install fastapi uvicorn pydantic openai openenv-core httpx
   ```

4. **Run tests**
   ```bash
   python test_local.py
   ```

---

### ✅ Solution 2: Install Rust (Advanced)

If you want to stick with Python 3.15 alpha and compile packages locally:

**Pros**: Future-proof for Python 3.15
**Time**: 15-20 minutes (includes Rust installation)

#### Steps:

1. **Install Rust from https://rustup.rs/**
   - Windows: Download installer, run
   - macOS/Linux: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

2. **Verify installation**
   ```bash
   rustc --version
   cargo --version
   ```

3. **Install Python dependencies**
   ```bash
   pip install fastapi uvicorn pydantic openai openenv-core httpx
   ```
   (This will now compile pydantic-core successfully)

4. **Run tests**
   ```bash
   python test_local.py
   ```

---

### ✅ Solution 3: Use Pre-compiled Docker Image

**Pros**: No local compilation, reproducible environment
**Time**: 10 minutes

#### Steps:

1. **Build Docker image**
   ```bash
   cd dynamic_pricing_env
   docker build -t pricing_env:latest .
   ```

2. **Run container**
   ```bash
   docker run -p 7860:7860 pricing_env:latest
   ```

3. **Access server**
   - Navigate to `http://localhost:7860`

---

## Recommendation

**For immediate progress**: Use **Solution 1 (Python 3.11)**
- Fastest setup
- All wheels pre-built
- Works immediately
- Sufficient for hackathon submission

**For long-term**: Consider **Solution 2 (Install Rust)** once Python 3.15 stable releases have wheel support.

---

## Version Requirements

Current project requires:
- **Python**: ≥3.10 (optimized for 3.11+)
- **FastAPI**: ≥0.110.0
- **Uvicorn**: ≥0.29.0
- **Pydantic**: ≥2.0.0
- **OpenEnv-core**: ≥0.2.1
- **OpenAI**: ≥1.0.0

All pre-compiled wheels available for Python 3.11, 3.12, 3.13.
Python 3.15 wheels not yet widely available (alpha status).

---

## Quick Commands Reference

```bash
# Create venv (Python 3.11)
python3.11 -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
# or
pip install fastapi uvicorn pydantic openai openenv-core httpx

# Run demo (no dependencies needed)
python demo_environment.py

# Run tests
python test_local.py

# Start server
python -m uvicorn server.app:app --reload --port 7860

# Check installed packages
pip list

# Verify environment
python -c "import openenv, fastapi, pydantic; print('All imports OK!')"
```

---

## After Setup: Verify Everything Works

```python
# test_setup.py
import sys
print(f"Python version: {sys.version}")

# Test imports
try:
    import fastapi
    print(f"✓ FastAPI {fastapi.__version__}")
except ImportError as e:
    print(f"✗ FastAPI: {e}")

try:
    import pydantic
    print(f"✓ Pydantic {pydantic.__version__}")
except ImportError as e:
    print(f"✗ Pydantic: {e}")

try:
    import openenv
    print(f"✓ OpenEnv Core")
except ImportError as e:
    print(f"✗ OpenEnv Core: {e}")

try:
    from server.pricing_environment import DynamicPricingEnvironment
    print(f"✓ DynamicPricingEnvironment imports successfully")
except ImportError as e:
    print(f"✗ DynamicPricingEnvironment: {e}")

try:
    from models import PricingAction, PricingObservation
    print(f"✓ Models import successfully")
except ImportError as e:
    print(f"✗ Models: {e}")

print("\n✓ All checks passed! Environment is ready.")
```

**Run it**:
```bash
python test_setup.py
```

---

## Environment Variable Configuration (Optional)

For the inference script, set these if using OpenAI:

```bash
# For bash/zsh
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4"
export SPACE_URL="http://localhost:7860"
export HF_TOKEN="hf_..."

# For Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
$env:MODEL_NAME="gpt-4"
$env:SPACE_URL="http://localhost:7860"
$env:HF_TOKEN="hf_..."

# For Windows CMD
set OPENAI_API_KEY=sk-...
set MODEL_NAME=gpt-4
set SPACE_URL=http://localhost:7860
set HF_TOKEN=hf_...
```

---

## Still Having Issues?

Check:
1. Python version: `python --version` (should be 3.11+)
2. Pip updated: `pip --version` (upgrade if old: `pip install --upgrade pip`)
3. Virtual environment activated: `which python` (should show path inside venv)
4. Network: `pip install requests` (test pip connectivity)

If all else fails, try:
```bash
pip install --upgrade --force-reinstall fastapi pydantic openenv-core
```

---

## Summary

| Step | Command | Time |
|------|---------|------|
| 1. Install Python 3.11 | `python3.11 --version` | 5 min |
| 2. Create venv | `python3.11 -m venv venv` | 1 min |
| 3. Activate venv | `source venv/bin/activate` | Instant |
| 4. Install deps | `pip install -r requirements.txt` | 3 min |
| 5. Verify setup | `python test_setup.py` | 1 min |
| 6. Run demo | `python demo_environment.py` | 2 min |
| **Total** | | **12 minutes** |

Once done, you can:
- ✅ Run `python test_local.py` — validate all 3 tasks
- ✅ Run `python -m uvicorn server.app:app --reload` — start WebSocket server
- ✅ Run `python inference.py` — test LLM baseline agent
- ✅ Submit to HuggingFace Spaces — deploy your solution

Good luck! 🚀
