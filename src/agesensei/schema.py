"""Core data models shared across all agents."""

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """A scientific paper retrieved from literature search."""

    pmid: str | None = None
    doi: str | None = None
    title: str
    authors: list[str] = Field(default_factory=list)
    journal: str = ""
    year: int | None = None
    abstract: str = ""
    url: str = ""
    citation_count: int = 0
    relevance_score: float = 0.0

    source: str = "pubmed"  # pubmed | s2 | arxiv | pmc
    arxiv_id: str | None = None
    pmc_id: str | None = None
    categories: list[str] = Field(default_factory=list)  # arXiv categories

    sections: dict[str, str] = Field(default_factory=dict)
    full_text_fetched: bool = False


class Target(BaseModel):
    """A potential drug target extracted from literature."""

    gene_symbol: str  # e.g. "TP53", "SIRT1"
    gene_name: str = ""  # e.g. "Tumor protein p53"
    uniprot_id: str = ""  # e.g. "P04637"
    organism: str = "Homo sapiens"

    # Evidence from literature
    papers: list[Paper] = Field(default_factory=list)
    aging_relevance: str = ""  # LLM-generated summary
    mechanism: str = ""  # how it relates to aging

    # Extraction details
    role: str = ""  # role described in papers
    is_druggable: bool | None = None  # from extraction
    aging_link: str = ""  # aging relationship description
    paper_count: int = 0  # number of papers mentioning this target
    source_papers: list[str] = Field(default_factory=list)  # PMIDs/DOIs

    # HAGR cross-reference
    in_genage: bool = False
    in_cellage: bool = False

    # Scores (0-1)
    literature_score: float = 0.0  # strength of literature evidence
    druggability_score: float = 0.0  # how druggable is this target
    novelty_score: float = 0.0  # how novel (not yet extensively targeted)
    overall_score: float = 0.0  # weighted composite
    score: float = 0.0  # target extraction relevance score


class ProteinAnalysis(BaseModel):
    """ESM-2 analysis results for a protein."""

    gene_symbol: str
    uniprot_id: str = ""
    sequence: str = ""
    sequence_length: int = 0

    # ESM-2 results
    embedding_dim: int = 0
    mean_embedding: list[float] = Field(default_factory=list)

    # Mutation analysis
    sensitive_positions: list[dict] = Field(default_factory=list)  # high mutation effect positions
    conservation_scores: list[float] = Field(default_factory=list)

    # Structure (if ESMFold used)
    pdb_data: str = ""
    confidence_scores: list[float] = Field(default_factory=list)


class DrugInfo(BaseModel):
    """Known drug/compound information for a target."""

    drug_name: str
    drugbank_id: str = ""
    mechanism: str = ""
    status: str = ""  # approved, clinical, preclinical
    indication: str = ""


class DrugabilityAssessment(BaseModel):
    """Druggability evaluation for a target."""

    gene_symbol: str
    has_known_drugs: bool = False
    known_drugs: list[DrugInfo] = Field(default_factory=list)
    binding_site_count: int = 0
    protein_class: str = ""  # kinase, GPCR, etc.
    tractability: str = ""  # small molecule, antibody, etc.
    score: float = 0.0
    reasoning: str = ""


class PathwayInfo(BaseModel):
    """Pathway enrichment result."""

    pathway_id: str
    pathway_name: str
    source: str = ""  # KEGG, GO, Reactome
    p_value: float = 0.0
    genes_in_pathway: list[str] = Field(default_factory=list)
    aging_relevance: str = ""


class BaselineRow(BaseModel):
    """One row in an aging-intervention baseline comparison table.

    Extracted from Methods/Results sections of experimental papers.
    """

    intervention: str  # e.g. "rapamycin", "senolytic ABT-263"
    target_gene: str = ""  # e.g. "MTOR", "BCL-XL"
    organism: str = ""  # e.g. "C57BL/6 mouse", "C. elegans", "human"
    dose: str = ""  # e.g. "14 ppm in diet", "5 mg/kg i.p. 5d/mo"
    duration: str = ""  # e.g. "lifelong from 9 months"
    lifespan_delta_pct: float | None = None  # median lifespan change %
    max_lifespan_delta_pct: float | None = None
    healthspan_markers: list[str] = Field(default_factory=list)  # e.g. ["grip strength +15%"]
    adverse_effects: str = ""
    clinical_stage: str = ""  # preclinical / phase1 / phase2 / approved
    pmid: str = ""
    pmc_id: str = ""
    year: int | None = None
    notes: str = ""


class SectionExcerpt(BaseModel):
    """Single section read during a deep_read pass."""

    section: str  # canonical name (introduction / methods / results / ...)
    length_chars: int = 0  # how much was actually read
    summary: str = ""  # one-sentence LLM takeaway for this section (optional)


class PaperFinding(BaseModel):
    """Structured result of deep-reading a single paper against a question.

    DeepXiv-inspired: instead of returning raw sections, the agent picks the
    most relevant sections for the question, synthesizes bullet-point findings,
    and surfaces methods + limitations so a downstream agent (or human) can trust it.
    """

    pmid: str = ""
    doi: str = ""
    arxiv_id: str = ""
    title: str = ""
    year: int | None = None
    source: str = ""  # pubmed | s2 | arxiv

    question: str = ""
    relevance_score: float = 0.0  # 0.0 - 1.0
    key_findings: list[str] = Field(default_factory=list)
    methods_summary: str = ""
    limitations: str = ""
    best_quote: str = ""  # short evidence quote
    sections_read: list[SectionExcerpt] = Field(default_factory=list)
    read_mode: str = "deep"  # "brief" | "deep" (brief == abstract only fallback)


class DigestEntry(BaseModel):
    """One paper entry in a trending digest."""

    paper: Paper
    why_matters: str = ""  # one-line editorial hook
    deep_finding: PaperFinding | None = None  # populated only for top-K deep reads


class TrendingDigest(BaseModel):
    """Weekly trending-papers digest for an aging/longevity topic."""

    topic: str = ""
    period_start: str = ""
    period_end: str = ""
    generated_at: str = ""
    total_scanned: int = 0
    top_papers: list[DigestEntry] = Field(default_factory=list)


class StructurePrediction(BaseModel):
    """Protenix/AlphaFold3-class structure prediction result."""

    gene_symbol: str
    sequence: str = ""
    model_used: str = ""  # protenix_base_default_v1.0.0, protenix-mini, etc.
    cif_path: str = ""
    plddt_mean: float = 0.0  # per-residue confidence average (0-100)
    ptm: float = 0.0  # predicted TM-score (0-1)
    iptm: float = 0.0  # interface pTM for complexes (0-1)
    num_residues: int = 0
    prediction_time_sec: float = 0.0
    error: str | None = None


class DiscoveryReport(BaseModel):
    """Final output of a target discovery run."""

    query: str
    timestamp: str = ""
    total_papers_analyzed: int = 0
    targets: list[Target] = Field(default_factory=list)
    protein_analyses: dict[str, ProteinAnalysis] = Field(default_factory=dict)
    structure_predictions: dict[str, StructurePrediction] = Field(default_factory=dict)
    druggability: dict[str, DrugabilityAssessment] = Field(default_factory=dict)
    pathways: list[PathwayInfo] = Field(default_factory=list)
    baseline_tables: dict[str, list[BaselineRow]] = Field(default_factory=dict)  # topic -> rows
    findings: list[PaperFinding] = Field(default_factory=list)  # deep-read per-paper findings
    summary: str = ""  # LLM-generated executive summary
