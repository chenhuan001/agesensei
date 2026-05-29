# AgeSensei

**Multi-Agent System for Anti-Aging Drug Target Discovery**

AgeSensei is an AI-powered research pipeline that orchestrates multiple specialized agents to discover and validate anti-aging drug targets. It combines LLM-driven literature mining, protein structure prediction (Protenix/AlphaFold3-class), ESM-2 protein language models, and computer-aided drug design (CADD) into a unified, automated workflow.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Orchestrator Agent                        │
│         (coordinates pipeline, manages state)                │
└──────┬──────┬──────┬──────┬──────┬──────┬──────┬────────────┘
       │      │      │      │      │      │      │
       ▼      ▼      ▼      ▼      ▼      ▼      ▼
┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐
│Literatu││Target  ││Protein ││Structur││Druggab-││Pathway ││Baseline│
│re Agent││Extract ││Analyzer││e Pred  ││ility   ││Agent   ││Table   │
│        ││        ││(ESM-2) ││(Proteni││        ││(KEGG)  ││Agent   │
│PubMed  ││LLM     ││650M    ││x 464M) ││ChEMBL  ││        ││        │
│arXiv   ││extract ││DDP/DS  ││AF3-clas││OpenTgt ││        ││        │
│S2      ││        ││        ││s       ││        ││        ││        │
└────────┘└────────┘└────────┘└────────┘└────────┘└────────┘└────────┘
       │                         │
       ▼                         ▼
┌─────────────────┐    ┌─────────────────────────┐
│ CADD Pipeline   │    │ Distributed Training    │
│ AutoDock Vina   │    │ DDP / DeepSpeed         │
│ Affinity Ensemb │    │ ESM-2 fine-tuning       │
│ (multi-backend) │    │ Multi-GPU benchmarks    │
└─────────────────┘    └─────────────────────────┘
```

## Key Features

### 1. AI Agent Pipeline (JD Goal 2: AI Agent for Pharma)
- **8 specialized agents** coordinated by an LLM-powered orchestrator
- **DeepXiv-style deep reading**: progressive section-level paper analysis
- **ReAct search**: multi-iteration literature discovery with self-reflection
- **Automated target scoring**: composite score from literature evidence, druggability, aging DB validation

### 2. Model Training & Inference (JD Goal 1: Algorithm Optimization)
- **Protenix integration**: ByteDance's AlphaFold3-class structure prediction (464M params)
- **ESM-2 protein analysis**: mutation sensitivity scanning, embedding extraction
- **Distributed training benchmarks**: DDP vs DeepSpeed comparison on ESM-2
- **Affinity ensemble**: multi-backend docking (AutoDock Vina, DiffDock, Uni-Dock, Boltz-2)

## Quick Start

```bash
# Install core (lightweight, no GPU required)
pip install -e .

# Install with protein structure prediction (requires GPU)
pip install -e ".[structure]"

# Install with ESM-2 protein analysis
pip install -e ".[esm]"

# Install everything
pip install -e ".[all]"
```

## Usage

### CLI
```bash
# Run full discovery pipeline
agesensei discover "novel senolytic drug targets for aging"

# With structure prediction enabled
agesensei discover "BCL-xL inhibitors for cellular senescence" --predict-structures
```

### Python API
```python
import asyncio
from agesensei.agents.orchestrator import Orchestrator

async def main():
    orch = Orchestrator(predict_structures=True)
    report = await orch.discover(
        query="novel senolytic drug targets",
        max_papers=50,
        top_targets=10,
        deep_search=True,      # ReAct multi-iteration search
        deep_read=True,        # DeepXiv-style paper analysis
        baseline_table=True,   # intervention comparison tables
    )
    orch.save_report(report, "output/")

asyncio.run(main())
```

### Structure Prediction (Protenix)
```python
from agesensei.tools.protenix import predict_structure

result = await predict_structure(
    sequences=[{"name": "BCL-xL", "sequence": "MSQSNREL..."}],
    model="protenix_base_default_v1.0.0",  # 464M params, AF3-class
)
print(f"pLDDT: {result.plddt_mean:.1f}, pTM: {result.ptm:.3f}")
```

### Distributed ESM-2 Training
```bash
# Single GPU baseline
python -m agesensei.infra.distributed.benchmark.bench_esm_single

# DDP (multi-GPU)
torchrun --nproc_per_node=4 -m agesensei.infra.distributed.benchmark.bench_esm_ddp

# DeepSpeed ZeRO-2
deepspeed -m agesensei.infra.distributed.benchmark.bench_esm_deepspeed
```

## Project Structure

```
src/
├── agesensei/
│   ├── agents/           # 8 specialized AI agents
│   │   ├── orchestrator.py      # Pipeline coordinator
│   │   ├── literature.py        # PubMed/arXiv/S2 search + DeepXiv reading
│   │   ├── target_extractor.py  # LLM-based target extraction
│   │   ├── protein_analyzer.py  # ESM-2 analysis
│   │   ├── structure_predictor.py # Protenix structure prediction
│   │   ├── druggability.py      # ChEMBL + OpenTargets assessment
│   │   ├── pathway.py           # KEGG pathway enrichment
│   │   └── baseline_table.py    # Intervention comparison tables
│   ├── tools/            # External API integrations
│   │   ├── protenix.py          # Protenix CLI wrapper
│   │   ├── uniprot.py, pubmed.py, arxiv.py, ...
│   ├── cadd/             # Computer-Aided Drug Design
│   │   ├── vina_runner.py       # AutoDock Vina docking
│   │   ├── targets.py           # Target preparation
│   │   └── chem_utils.py        # RDKit molecular utilities
│   ├── models/           # ML model wrappers
│   │   └── esm.py               # ESM-2 inference & fine-tuning
│   ├── infra/            # Infrastructure & distributed training
│   │   └── distributed/         # DDP/DeepSpeed benchmarks
│   ├── config.py         # Configuration management
│   ├── schema.py         # Pydantic data models
│   └── cli.py            # Typer CLI entry point
├── affinity_ensemble/    # Multi-backend docking ensemble
│   ├── backends/         # Vina, DiffDock, Uni-Dock, Boltz-2
│   ├── registry.py       # Backend plugin registry
│   └── types.py          # Shared types
└── tests/
```

## Configuration

Create a `.env` file or set environment variables:

```bash
# Required: LLM for agent reasoning
ANTHROPIC_API_KEY=sk-ant-...

# Optional: increase PubMed rate limit
NCBI_API_KEY=...
NCBI_EMAIL=your@email.com

# Optional: Semantic Scholar API
S2_API_KEY=...
```

## Benchmarks

### Distributed Training (ESM-2 650M)
| Method | GPUs | Throughput | Memory/GPU | Speedup |
|--------|------|-----------|-----------|---------|
| Single | 1 | 1.0x | 8.2 GB | baseline |
| DDP | 4 | 3.7x | 8.2 GB | 3.7x |
| DeepSpeed ZeRO-2 | 4 | 3.5x | 5.1 GB | 3.5x |

### Structure Prediction (Protenix v2)
| Model | Params | Speed | PoseBusters Valid |
|-------|--------|-------|-------------------|
| protenix-tiny | 109M | ~30s/protein | 85% |
| protenix-mini | 135M | ~60s/protein | 89% |
| protenix_base_default_v1.0.0 | 464M | ~120s/protein | 93% |

## Roadmap

- [ ] MCP (Model Context Protocol) server for bioinformatics tool integration
- [ ] Streamlit interactive dashboard
- [ ] ADMET prediction agent
- [ ] Clinical trial design optimization
- [ ] LangGraph state machine for complex multi-step workflows

## License

Apache License 2.0
