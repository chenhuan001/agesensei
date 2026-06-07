# AgeSensei

**Multi-Agent System for Anti-Aging Drug Target Discovery**

AgeSensei is an AI-powered research pipeline that orchestrates multiple specialized agents to discover and validate anti-aging drug targets. It combines LLM-driven literature mining, protein structure prediction (Protenix/AlphaFold3-class), ESM-2 protein language models with domain-adaptive fine-tuning, and computer-aided drug design (CADD) into a unified, automated workflow.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          Orchestrator Agent                              │
│               (coordinates 8-step pipeline, manages state)               │
└───┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬─────────────────────┘
    │      │      │      │      │      │      │      │
    ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼
┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐
│Litera││Target││Protei││Struct││ CADD ││Drugga││Pathwa││Baseli│
│ture  ││Extra-││n Ana-││Predi-││Agent ││bility││y     ││ne    │
│Agent ││ctor  ││lyzer ││ct    ││      ││Agent ││Agent ││Table │
│      ││      ││(ESM2)││Agent ││ChEMBL││      ││(KEGG)││Agent │
│PubMed││LLM   ││650M  ││Prote-││Vina  ││Open- ││      ││      │
│arXiv ││NER   ││DDP/DS││nix   ││QSAR  ││Tgt   ││      ││      │
│S2    ││      ││      ││464M  ││RDKit ││      ││      ││      │
└──────┘└──────┘└──────┘└──┬───┘└──┬───┘└──────┘└──────┘└──────┘
                           │       │
                     structure → docking
                    (CIF/PDB)   (receptor)

Pipeline: Literature → Targets → Protein → Structure → CADD → Druggability → Pathway → Report
```

## Key Features

### Multi-Agent Pipeline
- **9 specialized agents** coordinated by an LLM-powered orchestrator (literature → target extraction → protein analysis → **structure prediction** → **CADD virtual screening** → druggability → pathway → baseline table → report)
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
│   ├── eval/            # Benchmark evaluation
│   │   └── lab_bench_adapter.py  # LAB-Bench adapter
│   ├── config.py         # Configuration management
│   ├── schema.py         # Pydantic data models
│   └── cli.py            # Typer CLI entry point
├── affinity_ensemble/    # Multi-backend docking ensemble
│   ├── backends/         # Vina, DiffDock, Uni-Dock, Boltz-2
│   ├── registry.py       # Backend plugin registry
│   └── types.py          # Shared types
└── tests/
```

## Evaluation (LAB-Bench)

AgeSensei includes a [LAB-Bench](https://github.com/Future-House/LAB-Bench) adapter for evaluating agent performance on biology research tasks. The adapter routes questions to domain-specific AgeSensei tools (PubMed/UniProt retrieval, ESM-2 analysis) and measures the impact of tool augmentation vs baseline LLM.

### Supported Eval Categories

| Category | Questions | AgeSensei Tool Augmentation |
|----------|-----------|----------------------------|
| LitQA2 | 199 | LiteratureAgent (PubMed + Semantic Scholar) |
| DbQA | ~200 | UniProt / ChEMBL database retrieval |
| SeqQA | ~200 | ESM-2 protein analysis |
| SuppQA | ~200 | Literature retrieval |
| ProtocolQA | ~200 | Literature retrieval |
| FigQA | ~200 | LLM-only (visual reasoning) |
| TableQA | ~200 | LLM-only (table reasoning) |

### Running Evaluations

```bash
# Install eval dependencies
pip install -e ".[eval]"

# Quick test (10 questions per eval)
agesensei eval-lab-bench --max-questions 10

# Full evaluation on LitQA2
agesensei eval-lab-bench --evals LitQA2 --model claude-sonnet-4-20250514

# Ablation study: with vs without AgeSensei tools
agesensei eval-lab-bench --ablation --max-questions 50

# Baseline (no tool augmentation)
agesensei eval-lab-bench --no-tools --evals LitQA2 DbQA SeqQA
```

### Benchmark Results (LitQA2)

| Version | Model | Accuracy | Key Technique | Gap to SOTA |
|---------|-------|----------|---------------|-------------|
| v1 | Haiku | 26% (13/50) | Single-token, no CoT | -40% |
| v2 | Haiku | 34% (17/50) | + Chain-of-Thought | -32% |
| v3 | Opus 4.6 | 52% (26/50) | + Model upgrade | -14% |
| v4 | Opus 4.6 | 56% (28/50) | + DOI full-text retrieval | -10% |
| v5 | Opus 4.6 | 58% (29/50) | + BM25 DocIndex | -8% |
| v6 | Opus 4.6 | 62% (31/50) | + Embedding hybrid + Multi-query + Few-shot | -4% |
| v7 | Opus 4.6 | 64% (32/50) | + Self-Consistency + Reflection + F-cal | -2% |
| v9 | Opus 4.6 | 68% (68/100) | + Agentic ReAct + WebSearch + S2 API | +2% |
| v10 | Opus 4.6 | 66% (66/100) | + MemPalace local vector store | ±0% |
| v12 | Opus 4.6 | 72% (36/50†) | + Selenium full-text crawling (84% coverage) | +6% |
| **v14** | **Opus 4.6** | **75% (75/100)** | **+ Full-text补齐 + DOI直读 + 多源爬取 + retry** | **+9%** |

**Comparison with SOTA:**

| System | Model | LitQA2 Accuracy | Architecture |
|--------|-------|----------------|-------------|
| **AgeSensei v14** | **Opus 4.6** | **75%** | **Agentic ReAct + MemPalace + DOI full-text + Selenium crawling + Self-Consistency + Reflection** |
| AgeSensei v9 | Opus 4.6 | 68% | Agentic ReAct (8-turn) + DOI full-text + WebSearch + BM25/Embedding hybrid |
| PaperQA2 | GPT-4o | 66% | Full-text RAG + iterative search + LLM reranking |
| Raw LLM | Opus 4.6 | 52% | No tools (multiple-choice, CoT) |
| Raw LLM | GPT-4o | ~25% | No tools (open-answer format) |

**Key findings:**
- **Full-text is the sole bottleneck**: In v14, questions with full-text available scored **100% (75/75)**, while questions without full-text scored **0% (0/25)** — model reasoning is not the limiting factor
- **Full-text coverage drives accuracy**: Every 1% improvement in full-text coverage translates directly to ~1% accuracy gain
- **Agentic ReAct is the breakthrough**: Letting the model autonomously decide what to search, which papers to read, and when to stop yields +16% over static retrieval
- **Model scale is the biggest lever**: Opus 4.6 baseline (52%) is 2x Haiku baseline (26%)
- **Self-Consistency + Reflection**: 3x majority voting + answer verification adds +4%
- **MemPalace local retrieval**: Pre-indexed 356 papers (26K drawers) enables fast offline retrieval
- **Exceeds SOTA by 9%**: AgeSensei v14 75% > PaperQA2 66% on LitQA2 (n=100)

**Error analysis (v14, 25 wrong answers):**

| Full-text Status | Questions | Accuracy | Notes |
|-----------------|-----------|----------|-------|
| **Full-text available (>5KB)** | 75 | **100%** | Zero errors when source paper is accessible |
| No full-text | 25 | 0% | All errors due to missing source papers |

Missing papers by publisher: bioRxiv (7, Cloudflare anti-bot), Elsevier/Cell (6, paywall), Nature (4, paywall), Science/AAAS (1), Others (7)

### Python API

```python
from agesensei.eval import run_lab_bench

results = await run_lab_bench(
    evals=["LitQA2", "DbQA", "SeqQA"],
    model="claude-sonnet-4-20250514",
    use_tools=True,
    n_threads=4,
)
for name, r in results.items():
    print(f"{name}: {r.accuracy:.1%} ({r.correct}/{r.total})")
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

## MemPalace Integration (Local Knowledge Store)

AgeSensei supports [MemPalace](https://github.com/MemPalace/mempalace) as a local vector store for pre-indexed paper retrieval. This eliminates runtime API dependencies and provides reproducible, fast evaluation.

```bash
# Install MemPalace
python -m venv ~/.mempalace-venv && ~/.mempalace-venv/bin/pip install mempalace

# Initialize palace for papers
~/.mempalace-venv/bin/mempalace init agesensei_papers

# Mine all cached full-text papers
~/.mempalace-venv/bin/mempalace mine artifacts/papers_text/ --palace agesensei_papers

# Search (used by eval adapter)
~/.mempalace-venv/bin/mempalace recall "H3.3K36R Drosophila eclosion" --palace agesensei_papers
```

**Stats:** 356 papers indexed → 26,308 semantic drawers → sub-second local retrieval

| Mode | Accuracy | Speed | API Dependency |
|------|----------|-------|----------------|
| **Agentic + MemPalace (v14)** | **75%** | ~2 min/question | LLM API only |
| Agentic (v9) | 68% | ~3 min/question | PubMed, S2, Unpaywall |
| MemPalace only (v10) | 66% | ~45 sec/question | None (fully offline) |

## Roadmap

- [x] LAB-Bench evaluation adapter (LitQA2, DbQA, SeqQA + tool augmentation ablation)
- [x] MemPalace local vector store integration (269 papers, 20K drawers)
- [ ] BixBench agentic evaluation (computational biology data analysis capsules)
- [ ] MCP (Model Context Protocol) server for bioinformatics tool integration
- [ ] Streamlit interactive dashboard
- [ ] ADMET prediction agent
- [ ] Clinical trial design optimization
- [ ] Integration with DMS (Deep Mutational Scanning) datasets for fine-tuning
- [ ] Active learning loop: pipeline discovers targets → fine-tune → better predictions

## License

Apache License 2.0
