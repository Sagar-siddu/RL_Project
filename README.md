# 🏪 Dynamic Pricing RL Environment

**A production-ready reinforcement learning environment for e-commerce SKU pricing optimization.**

---

## 🎯 Problem Statement

9.7 million Amazon sellers and 2+ million Shopify merchants face the same challenge daily: **How do I price my SKUs to maximize revenue without racing to the bottom or leaving money on the table?**

Existing solutions are limited:
- **Repricers** (Feedvisor, Profit Cyclops) are rule-based, not learning-based
- **Analytics tools** (Helium 10, Sellerboard) show numbers but take no action
- **Handmade strategies** don't scale to thousands of SKUs across dynamic markets

This environment fills the gap: a **realistic, multi-step RL environment** where agents learn jointly optimize revenue, inventory health, and competitive positioning.

---

## ✨ Key Features

✅ **Realistic Market Simulation**
- Price elasticity model with competitor response
- Inventory dynamics (stockouts, overstocking penalties)
- Time-varying demand signals

✅ **Multi-Episode Support**
- 50+ concurrent episodes simultaneously
- UUID-based episode tracking
- Full state isolation

✅ **3 Task Variants**
1. **single_sku_stable** — Single SKU, stable demand (baseline)
2. **multi_sku_competitors** — 5 SKUs with reactive competitors
3. **demand_shocks_perishables** — Volatile demand, inventory decay

✅ **Production Deployment**
- FastAPI server (async, scalable)
- Docker containerized
- OpenEnv-compliant (standard gym-like API)
- HuggingFace Spaces ready

---

## 🚀 Quick Start (5 minutes)

### Local Demo

1. **Clone & Install**
   ```bash
   git clone https://github.com/YOUR_USERNAME/RL_Project.git
   cd RL_Project
   pip install -r requirements.txt
   ```

2. **Start API Server**
   ```bash
   cd dynamic_pricing_env
   python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
   ```

3. **Run Interactive UI** (new terminal)
   ```bash
   python app.py
   ```
   Visit `http://localhost:7860` to see the Gradio interface

### Docker Deployment

```bash
docker build -t dynamic-pricing .
docker run -p 7860:7860 dynamic-pricing
```

### HuggingFace Spaces

Visit: `https://huggingface.co/spaces/YOUR_USERNAME/dynamic-pricing-rl`

---

## 📊 API Examples

### Create Episode

```bash
curl -X POST http://localhost:7860/episode/create \
  -H "Content-Type: application/json" \
  -d '{"task_name": "single_sku_stable"}'
```

**Response:**
```json
{
  "episode_id": "abc12345",
  "task_name": "single_sku_stable",
  "observation": {
    "prices": [50.0],
    "inventory": [300],
    "revenue": 0.0,
    "done": false
  },
  "reward": 0.0
}
```

### Execute Step

```bash
curl -X POST http://localhost:7860/episode/abc12345/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "prices": [52.50]
    }
  }'
```

### Grade Episode

```bash
curl -X POST http://localhost:7860/episode/abc12345/grade
```

---

## 🧪 Testing

Run the test suite:

```bash
# API endpoint tests (6/6 passing)
python test_api_endpoints.py

# Load tests (5/5 passing - 50 concurrent episodes)
python load_test.py
```

---

## 📈 Performance Benchmarks

| Metric | Value |
|--------|-------|
| Episode Creation | 5-50ms avg |
| Step Execution | 7-57ms avg |
| Max Concurrent Episodes | 50+ |
| Single-Sku Task Score | 0.95+ |

---

## 📚 Documentation

- [QUICK_START.md](./QUICK_START.md) — Installation & basic usage
- [SETUP_GUIDE.md](./SETUP_GUIDE.md) — Development environment setup
- [PROJECT_SUMMARY.md](./PROJECT_SUMMARY.md) — Architecture overview
- [dynamic_pricing_env/README.md](./dynamic_pricing_env/README.md) — Environment details
- [dynamic_pricing_env/COMPLETE_GUIDE.md](./dynamic_pricing_env/COMPLETE_GUIDE.md) — Full technical guide

---

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│   Gradio UI (app.py)                │
│   Interactive demo interface        │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│   FastAPI Server                    │
│   - Multi-episode support           │
│   - UUID-based tracking             │
│   - 50+ concurrent episodes         │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│   DynamicPricingEnvironment         │
│   - Market simulation               │
│   - Price elasticity model          │
│   - Competitor agents               │
│   - Reward shaping                  │
└─────────────────────────────────────┘
```

---

## 🎮 Environment API

### Action Space
```python
{"prices": [50.0, 60.5, 75.0]}  # one price per SKU, >0
```

### Observation Space
```python
{
  "prices": [50.0, 60.5, 75.0],      # current prices
  "inventory": [245, 180, 320],      # units in stock
  "competitor_prices": [51.0, 61.0, 74.5],
  "demand": [9.5, 8.2, 11.3],       # units demanded
  "revenue": 2450.50,                # cumulative episode revenue
  "done": false,                     # episode finished
  "reward": 0.92                     # shaped reward signal
}
```

### Reward Function

$$\text{reward} = w_1 \cdot \text{normalized\_revenue} - w_2 \cdot \text{stockout\_penalty} - w_3 \cdot \text{overstock\_penalty}$$

Where:
- $w_1 = 1.0$ (revenue weight)
- $w_2 = 0.5$ (stockout penalty)
- $w_3 = 0.3$ (overstock penalty)

---

## 🔧 Configuration

Set environment variables to customize:

```bash
export PRICING_TASK=multi_sku_competitors  # Task name
export LOG_LEVEL=INFO                      # Logging level
```

Available tasks:
- `single_sku_stable` (default)
- `multi_sku_competitors`
- `demand_shocks_perishables`

---

## 🏆 Hackathon Submission Checklist

- [x] Working FastAPI server (all endpoints tested)
- [x] Multi-episode support (50+ concurrent verified)
- [x] Interactive UI (Gradio demo)
- [x] Docker containerized
- [x] Full test suite (11/11 passing)
- [x] Comprehensive documentation
- [x] HuggingFace Spaces deployment ready

---

## 📝 Citation

If you use this environment in your research, please cite:

```bibtex
@software{dynamic_pricing_rl_2026,
  title={Dynamic Pricing RL Environment},
  author={Your Name},
  year={2026},
  url={https://github.com/YOUR_USERNAME/RL_Project}
}
```

---

## 📄 License

MIT License — See LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## 💬 Support

- **Issues:** [GitHub Issues](https://github.com/YOUR_USERNAME/RL_Project/issues)
- **Discussions:** [GitHub Discussions](https://github.com/YOUR_USERNAME/RL_Project/discussions)
- **Documentation:** See `/docs` folder

---

**Made with ❤️ for the Scaler Meta PyTorch Hackathon 2026**
