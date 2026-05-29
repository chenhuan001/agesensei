# AgeSensei

**Multi-Agent System for Anti-Aging Drug Target Discovery**

AgeSensei is an AI-powered research pipeline that orchestrates multiple specialized agents to discover and validate anti-aging drug targets. It combines LLM-driven literature mining, protein structure prediction (Protenix/AlphaFold3-class), ESM-2 protein language models with domain-adaptive fine-tuning, and computer-aided drug design (CADD) into a unified, automated workflow.

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
       │                         │                    │
       ▼                         ▼                    ▼
┌─────────────────┐    ┌─────────────────────────┐  ┌──────────────────┐
│ CADD Pipeline   │    │ ESM-2 Fine-tuning       │  │ Affinity Ensemble│
│ AutoDock Vina   │    │ Domain-adaptive MLM     │  │ Vina / DiffDock  │
│ Affinity Ensemb │    │ Mutation classification │  │ Uni-Dock / Boltz │
│ (multi-backend) │    │ DDP / DeepSpeed         │  │ (multi-backend)  │
└─────────────────┘    └─────────────────────────┘  └──────────────────┘
```

## Key Features

### Multi-Agent Pipeline
- **8 specialized agents** coordinated by an LLM-powered orchestrator
- **DeepXiv-style deep reading**: progressive section-level paper analysis
- **ReAct search**: multi-iteration literature discovery with self-reflection
- **Automated target scoring**: composite score from literature evidence, druggability, aging DB validation

### Model Training & Inference
- **Protenix integration**: AlphaFold3-class structure prediction (464M params)
- **ESM-2 protein analysis**: mutation sensitivity scanning, embedding extraction
- **Domain-adaptive fine-tuning**: MLM on GenAge proteins + mutation effect classification
- **Distributed training**: DDP and DeepSpeed ZeRO-2 support for multi-GPU fine-tuning
- **Affinity ensemble**: multi-backend docking (AutoDock Vina, DiffDock, Uni-Dock, Boltz-2)

### Fine-tuning Pipeline
The fine-tuning module enables domain adaptation of ESM-2 on aging-related proteins, improving mutation effect prediction accuracy for senescence-associated targets:

1. **Data preparation**: Fetches GenAge protein sequences + generates synthetic mutation labels
2. **MLM pre-training**: Masked language modeling on aging protein corpus (domain adaptation)
3. **Mutation classification**: Binary prediction of deleterious vs neutral mutations
4. **Pipeline integration**: Fine-tuned model automatically used by ProteinAnalyzer agent

## Quick Start

```bash
# Install core (lightweight, no GPU required)
pip install -e .

# Install with ESM-2 fine-tuning support
pip install -e ".[distributed]"

# Install with protein structure prediction (requires GPU)
pip install -e ".[structure]"

# Install everything
pip install -e ".[all]"
```

## Usage

### Full Discovery Pipeline
```bash
# Run full discovery pipeline
agesensei discover "novel senolytic drug targets for aging"

# With structure prediction enabled
agesensei discover "BCL-xL inhibitors for cellular senescence" --predict-structures

# With deep search and deep reading
agesensei discover "mTOR pathway aging interventions" --deep-search --deep-read
```

### ESM-2 Fine-tuning
```bash
# Step 1: Prepare training data from GenAge
agesensei prepare-finetune-data --output data/finetune --task both

# Step 2: Fine-tune with MLM (domain adaptation)
agesensei finetune data/finetune/train_mlm.csv --task mlm --epochs 5 --bf16

# Step 3: Fine-tune for mutation classification
agesensei finetune data/finetune/train_mutations.csv \
    --task mutation_cls --model artifacts/finetune/best --epochs 10

# Multi-GPU fine-tuning (DDP)
torchrun --nproc_per_node=4 -m agesensei.infra.distributed.finetune.trainer \
    --data data/finetune/train_mlm.csv --task mlm --epochs 5 --bf16

# DeepSpeed ZeRO-2
deepspeed -m agesensei.infra.distributed.finetune.trainer \
    --data data/finetune/train_mlm.csv --task mlm --epochs 5 --deepspeed --bf16
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
        deep_search=True,
        deep_read=True,
        baseline_table=True,
    )
    orch.save_report(report, "output/")

asyncio.run(main())
```

### Using Fine-tuned Model in Pipeline
```python
from agesensei.config import config

# Point to your fine-tuned checkpoint
config.esm.finetuned_path = "artifacts/finetune/best"

# Now all ESM-2 operations use the fine-tuned model
from agesensei.agents.protein_analyzer import ProteinAnalyzer
analyzer = ProteinAnalyzer()
result = await analyzer.analyze("SIRT1", scan_positions=50)
```

## Distributed Training Benchmarks

### ESM-2 650M Inference
| Method | GPUs | Throughput | Memory/GPU | Speedup |
|--------|------|-----------|-----------|---------|
| Single | 1 | 1.0x | 8.2 GB | baseline |
| DDP | 4 | 3.7x | 8.2 GB | 3.7x |
| DeepSpeed ZeRO-2 | 4 | 3.5x | 5.1 GB | 3.5x |

### ESM-2 650M Fine-tuning (MLM, batch=16, seq_len=512)
| Method | GPUs | Steps/s | Memory/GPU | Notes |
|--------|------|---------|-----------|-------|
| Single | 1 | 1.0x | 14.2 GB | gradient accumulation=4 |
| DDP | 4 | 3.6x | 14.2 GB | near-linear scaling |
| DeepSpeed ZeRO-2 | 4 | 3.4x | 9.8 GB | 31% memory reduction |

### Structure Prediction (Protenix v2)
| Model | Params | Speed | PoseBusters Valid |
|-------|--------|-------|-------------------|
| protenix-tiny | 109M | ~30s/protein | 85% |
| protenix-mini | 135M | ~60s/protein | 89% |
| protenix_base_default_v1.0.0 | 464M | ~120s/protein | 93% |

## Project Structure

```
src/
├── agesensei/
│   ├── agents/           # 8 specialized AI agents
│   │   ├── orchestrator.py      # Pipeline coordinator
│   │   ├── literature.py        # PubMed/arXiv/S2 search + DeepXiv reading
│   │   ├── target_extractor.py  # LLM-based target extraction
│   │   ├── protein_analyzer.py  # ESM-2 analysis (supports fine-tuned models)
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
│   │   └── esm.py               # ESM-2 inference (pre-trained + fine-tuned)
│   ├── infra/            # Infrastructure & distributed training
│   │   └── distributed/
│   │       ├── benchmark/       # DDP/DeepSpeed inference benchmarks
│   │       └── finetune/        # ESM-2 fine-tuning pipeline
│   │           ├── dataset.py       # AgingMutationDataset
│   │           ├── trainer.py       # ESMFineTuner (DDP/DeepSpeed)
│   │           └── prepare_data.py  # GenAge data preparation
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

# Optional: use fine-tuned ESM-2 model
ESM_FINETUNED_PATH=artifacts/finetune/best
```

## Roadmap

- [ ] MCP (Model Context Protocol) server for bioinformatics tool integration
- [ ] Streamlit interactive dashboard
- [ ] ADMET prediction agent
- [ ] Clinical trial design optimization
- [ ] LangGraph state machine for complex multi-step workflows
- [ ] Integration with DMS (Deep Mutational Scanning) datasets for fine-tuning
- [ ] Active learning loop: pipeline discovers targets → fine-tune → better predictions

## License

Apache License 2.0
