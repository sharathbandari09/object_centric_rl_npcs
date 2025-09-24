# Object-Centric Reinforcement Learning for NPCs: A Comprehensive Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Paper](https://img.shields.io/badge/Paper-Thesis-green.svg)](thesis%20docs/masters_thesis.pdf)

This research project investigates the boundaries between reinforcement learning and transformer-based reasoning systems for spatial intelligence tasks. The study demonstrates that **SLM-based program synthesis achieves 100% success with perfect generalization** while traditional RL methods fail to generalize beyond the KeyDoor training distribution (0% success on novel environments).

## 🎯 Core Research Question

**Can traditional RL methods match the spatial reasoning and generalization capabilities of transformer-based program synthesis approaches?**

Our findings prove that transformer architectures with natural language reasoning fundamentally outperform neural RL approaches in spatial reasoning domains.

## 📊 Key Results

| Method | Training Success | Novel Success | Generalization Gap |
|--------|------------------|---------------|-------------------|
| **SLM Program Synthesis** | **100%** | **100%** | **0%** |
| BC Raw/Abstract | 100% | 0% | 100% |
| MoE Raw/Abstract | 100% | 0% | 100% |
| PPO Raw/Abstract | 10-17% | 8-10% | ~85% |

## 🏗️ Project Structure

```
Masters_project/
├── agents/                          # RL Agent Implementations
│   ├── bc/                         # Behavioral Cloning agents
│   ├── moe/                        # Mixture of Experts agents
│   └── ppo/                        # PPO agents
├── datasets/                        # Training datasets and generation
├── dungeon_game/                    # Memory Maze: Exploratory 80×20 benchmarking pygame for future AI agents—80×20
├── envs/                           # KeyDoor environment implementations
├── experiments/                     # Trained model checkpoints and results
├── results/                        # Thesis visualizations and analysis
│   ├── figures/                    # Generated charts and graphs
│   └── data/                       # Experimental results data
├── scripts/                        # Training and testing scripts
│   ├── bc/                         # BC training/testing scripts
│   ├── moe/                        # MoE training/testing scripts
│   └── ppo/                        # PPO training/testing scripts
├── slm-program-synthesis_keydoor/   # SLM Implementation
├── sprites/                        # Game sprites and visual assets
├── templates/                      # Environment template definitions
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Git**
- **Ollama** (for SLM approach)

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/object-centric-rl-npcs.git
cd object-centric-rl-npcs
```

### 2. Install Core Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install torch torchvision numpy pygame gymnasium matplotlib seaborn pandas scipy
```

### 3. Quick Demo - SLM Program Synthesis (Recommended)

The **breakthrough approach** that achieves perfect generalization:

```bash
# Install Ollama (if not already installed)
# Visit: https://ollama.ai/

# Pull Llama 3.2 3B model
ollama pull llama3.2:3b

# Install SLM dependencies
cd slm-program-synthesis_keydoor/
pip install -r requirements.txt

# Run SLM demo
python slm_demo.py
```

**Expected Output**: 100% success rate with step-by-step spatial reasoning

## 📦 Installation Guide

### Option 1: SLM Program Synthesis Only (Recommended)

slm outperforms all RL methods:

```bash
cd slm-program-synthesis_keydoor/
pip install ollama>=0.1.0 numpy>=1.21.0 dataclasses-json>=0.5.0
ollama pull llama3.2:3b
python slm_demo.py
```

### Option 2: Complete Research Environment

For reproducing all experiments and comparisons:

```bash
# Install all dependencies
pip install -r requirements_full.txt

# Or install manually:
pip install torch>=1.9.0 torchvision numpy>=1.21.0 pygame>=2.0.0
pip install gymnasium matplotlib>=3.5.0 seaborn>=0.11.0 pandas scipy
pip install stable-baselines3 tensorboard

# For SLM approach
pip install ollama>=0.1.0 dataclasses-json>=0.5.0

# Install Ollama and pull model
ollama pull llama3.2:3b
```

## 🎮 Running Experiments

### SLM Program Synthesis (Perfect Performance)

```bash
cd slm-program-synthesis_keydoor/

# Quick demo
python slm_demo.py

# Full environment test
python test_slm_env.py  

# Visual pygame test
python test_slm_pygame.py
```

### Traditional RL Methods

#### Behavioral Cloning
```bash
cd scripts/bc/

# Train BC models
python train_bc_abstract.py
python train_bc_raw.py

# Test BC models  
python test_bc_abstract.py
python test_bc_raw.py
```

#### Mixture of Experts
```bash
cd scripts/moe/

# Train MoE models
python train_moe_abstract.py
python train_moe_raw.py

# Test MoE models
python test_moe_abstract.py
python test_moe_raw.py
```

#### PPO (Policy Gradient)
```bash
cd scripts/ppo/

# Train PPO models
python train_ppo_abstract.py
python train_ppo_raw.py

# Test PPO models
python test_ppo_abstract.py
python test_ppo_raw.py
```

## 🔬 Key Research Findings

### 1. **SLM Program Synthesis Breakthrough**
- **100% success rate** on both training and novel environments
- **Perfect generalization** with 0% generalization gap
- **Natural language reasoning** enables spatial intelligence
- **Ready-to-deploy** without training requirements

### 2. **Traditional RL Limitations**
- **Catastrophic generalization failure**: 0% success on novel environments
- **Object-centric representations provide no advantage** over raw pixels
- **Covariate shift** is the fundamental limiting factor
- **Perfect training performance** followed by complete failure

### 3. **Paradigm Shift Evidence**
- **Statistical learning (RL) vs. Symbolic reasoning (Transformers)**
- **Pre-trained world knowledge** beats learned patterns
- **Compositional understanding** enables true generalization
- **Natural language integration** provides spatial reasoning capabilities

### 4. **Memory Maze Exploratory Insights**
- **AI Evaluation Testbed**: An 80×20 grid-based dungeon crawler designed as a benchmark for assessing next-gen AI game agents' spatial intelligence, memory, and multi-modal capabilities—e.g., invisible mazes test long-term planning, password systems evaluate coordinate/math integration.
- **Thesis Insights for Future Work**: Documents baseline limitations (e.g., 35-80% VLM accuracy in coordinate recognition and command generation) to guide hybrid AI development; ideal for benchmarking RL, VLM, and SLM agents in complex, dynamic environments
- **Implementation Note**: Built as an 80×20 grid-based dungeon maze `dungeon_game/` - its a benchmarking game for future ai game agents to prototype and measure progress toward human-level game AI.

## 📁 Dataset Information

The research uses the **KeyDoor Environment** with:
- **10 templates** (T1-T6 training, T7-T10 novel)
- **Grid-based navigation** (6×6 to 12×12)
- **Spatial reasoning tasks**: collect keys, open doors
- **Perfect expert demonstrations** from oracle policy
- **5,100+ training observations**

## 🏆 Performance Benchmarks

### SLM Program Synthesis Performance
| Template | Type | Success | Steps | Response Time |
|----------|------|---------|-------|---------------|
| T1-T6 | Training | 100% | 16.5 avg | 3.4s avg |
| T7-T10 | Novel | 100% | 16.5 avg | 3.4s avg |

### Traditional RL Comparison
- **BC Methods**: 100% training → 0% novel
- **MoE Methods**: 100% training → 0% novel  
- **PPO Methods**: 10-17% training → 8-10% novel

## 🛠️ Troubleshooting

### Common Issues

**1. Ollama Connection Error**
```bash
# Ensure Ollama is running
ollama serve

# Check if model is available
ollama list
```

**2. Missing Dependencies**
```bash
# Install missing packages
pip install torch numpy pygame gymnasium matplotlib seaborn
```

**3. CUDA Issues (Optional)**
```bash
# For GPU support (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**4. Permission Issues**
```bash
# If file permission errors occur
chmod +x scripts/*.py
```

## 🎯 Usage Examples

### 1. Run SLM Demo
```python
from keydoor_slm_env import KeyDoorSLMEnv
from slm_agent import SLMKeyDoorAgent

env = KeyDoorSLMEnv(grid_size=8)
agent = SLMKeyDoorAgent(model_name="llama3.2:3b")

obs, info = env.reset(template_id=1)
action = agent.act(env)  # Returns optimal action with reasoning
```

### 2. Train RL Agent
```python
from agents.bc.bc_abstract import BCAbstract
from envs.keydoor.keydoor_env import KeyDoorEnv

env = KeyDoorEnv()
agent = BCAbstract()
# Training code...
```

### 3. Generate Visualizations
```python
from results.generate_thesis_visualizations import ThesisVisualizationGenerator

generator = ThesisVisualizationGenerator()
generator.generate_all_visualizations()
```

## 📈 Research Impact

This research demonstrates:

1. **Paradigm Boundaries**: Clear evidence of transformer superiority over RL
2. **Practical Applications**: Ready-to-deploy spatial reasoning without training
3. **Theoretical Insights**: Statistical learning vs. symbolic reasoning limitations
4. **Future Directions**: Hybrid approaches combining RL efficiency with transformer reasoning
5. **Benchmarking Framework**: Memory Maze delivers a custom-engineered puzzle game testbed for next-gen AI agents, offering  to evaluate spatial and multi-modal reasoning capabilities.

## 👥 Authors

- **Sharath Bandari** 

## 🙏 Acknowledgments

- Ollama team for the excellent LLM serving framework
- Llama 3.2 3B model for spatial reasoning capabilities
- Research community for foundational work in object-centric learning
- Open source contributors for the tools and libraries used

---

**🎓 Master's Thesis Research Project**  
**Bournemouth University | Year 2025**  
**Supervisor: Jon Macey**
