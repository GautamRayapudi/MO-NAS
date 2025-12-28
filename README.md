# MO-NAS: Multi-Objective Neural Architecture Search

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)

A production-ready framework for automatically discovering optimal neural network architectures using **NSGA-II multi-objective optimization**.

## ✨ Features

- 🎯 **Multi-Objective**: Accuracy, FLOPs, Parameters, Latency, Memory
- 📊 **Multi-Modal**: Image, Text, Sequence, Tabular data
- ⚡ **Zero-Cost Proxies**: Fast evaluation without training
- 🧠 **Bayesian Guidance**: Surrogate model for efficiency
- 🔄 **Weight Sharing**: Reduced training cost
- 🏗️ **Modular Design**: Easy to extend

---

## � How It Works

```mermaid
flowchart LR
    subgraph Input
        A[Dataset Config]
        B[Search Config]
    end
    
    subgraph NSGA-Net Search
        C[Initialize Population]
        D[Evaluate Architectures]
        E[NSGA-II Selection]
        F[Crossover & Mutation]
        G[Bayesian Guidance]
    end
    
    subgraph Output
        H[Pareto Front]
        I[Best Architecture]
        J[PyTorch Model]
    end
    
    A --> C
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> D
    E --> H
    H --> I
    I --> J
```

---

## 🎯 Multi-Objective Optimization

NSGA-Net finds the **Pareto-optimal** trade-off between competing objectives:

```mermaid
quadrantChart
    title Pareto Front: Accuracy vs Efficiency
    x-axis Low FLOPs --> High FLOPs
    y-axis Low Accuracy --> High Accuracy
    quadrant-1 High Acc, High Cost
    quadrant-2 Optimal Zone
    quadrant-3 Low Acc, Low Cost
    quadrant-4 Avoid
    
    Arch A: [0.8, 0.9]
    Arch B: [0.5, 0.85]
    Arch C: [0.3, 0.75]
    Arch D: [0.6, 0.7]
    Arch E: [0.2, 0.6]
```

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph nsga_net["📦 nsga_net Package"]
        subgraph core["🔧 core/"]
            C1[config.py<br/>DatasetConfig, SearchConfig]
            C2[architecture.py<br/>Architecture, LayerConfig]
            C3[model_builder.py<br/>UniversalModelBuilder]
        end
        
        subgraph algorithms["⚙️ algorithms/"]
            A1[nsga.py<br/>NSGA-II Algorithm]
            A2[genetic_ops.py<br/>Crossover, Mutation]
            A3[bayesian.py<br/>BayesianGuidance]
        end
        
        subgraph evaluation["📊 evaluation/"]
            E1[proxies.py<br/>ZeroCostProxy]
            E2[trainer.py<br/>Weight Sharing, Training]
        end
        
        subgraph search["🔍 search/"]
            S1[search_spaces.py<br/>Image, Text, Sequence, Tabular]
            S2[nsga_net.py<br/>Main NSGANet Class]
        end
        
        subgraph utils["🛠️ utils/"]
            U1[analysis.py<br/>ResultsAnalyzer]
        end
    end
    
    S2 --> A1
    S2 --> A2
    S2 --> A3
    S2 --> E2
    S2 --> S1
    E2 --> E1
    E2 --> C3
    S1 --> C2
    C2 --> C1
```

---

## 📦 Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🚀 Quick Start

```python
from nsga_net import DatasetConfig, SearchConfig, NSGANet, UniversalModelBuilder

# Configure
dc = DatasetConfig('image', (3, 224, 224), 1000)
sc = SearchConfig(population_size=20, generations=30)

# Search
nsga = NSGANet(dc, sc)
pareto = nsga.search(train_loader, val_loader)

# Get best & build model
best = nsga.get_best_architecture(pareto, 'balanced')
model = UniversalModelBuilder.build_model(best)
```

---

## 🔬 Search Process

```mermaid
sequenceDiagram
    participant User
    participant NSGANet
    participant SearchSpace
    participant Trainer
    participant NSGAII
    
    User->>NSGANet: search(train_loader, val_loader)
    
    loop For each generation
        NSGANet->>SearchSpace: sample_architecture()
        SearchSpace-->>NSGANet: Architecture configs
        
        NSGANet->>Trainer: train_architecture()
        Note over Trainer: Zero-cost proxy OR<br/>Weight-shared training
        Trainer-->>NSGANet: Accuracy, FLOPs, Params
        
        NSGANet->>NSGAII: non_dominated_sort()
        NSGAII-->>NSGANet: Pareto fronts
        
        NSGANet->>NSGAII: select_population()
        Note over NSGAII: Crowding distance<br/>for diversity
    end
    
    NSGANet-->>User: Pareto-optimal architectures
```

---

## 🎨 Supported Data Types

| Type | Input Shape | Search Space | Examples |
|------|-------------|--------------|----------|
| 🖼️ **Image** | `(C, H, W)` | Conv, Pool, Skip | CIFAR-10, ImageNet |
| 📝 **Text** | `seq_length` | LSTM, GRU, Transformer | Sentiment, NER |
| 📈 **Sequence** | `(seq, features)` | RNN, Dense | Time Series, Stock |
| 📊 **Tabular** | `num_features` | Dense, Dropout, BN | Credit, Fraud |

---

## ⚙️ Configuration

```python
SearchConfig(
    population_size=20,         # Number of architectures per generation
    generations=30,             # Evolution iterations
    max_flops=1000,             # FLOPs limit (millions)
    max_memory_mb=8000,         # GPU memory limit
    use_zero_cost_proxy=True,   # Fast evaluation
    use_bayesian_guidance=True, # Surrogate model
)
```

---

## 📁 Project Structure

```
MO-NAS/
├── nsga_net/                    # Main package
│   ├── __init__.py             # Public API
│   ├── core/                    # Core components
│   ├── algorithms/              # Optimization algorithms
│   ├── evaluation/              # Evaluation utilities
│   ├── search/                  # Search components
│   └── utils/                   # Utilities
├── main.py                      # Demo entry point
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

---

## 🏃 Run Demo

```bash
python main.py
```

## 📤 Export Results

```python
from nsga_net import ResultsAnalyzer

ResultsAnalyzer.export_architecture(best, 'best_arch.json')
ResultsAnalyzer.save_search_results(nsga, pareto, 'results.json')
```

---

**Happy Architecture Searching! 🚀**
