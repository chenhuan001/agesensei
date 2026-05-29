"""Tests for core data models."""

from agesensei.schema import Paper, Target, ProteinAnalysis, DiscoveryReport


def test_paper_creation():
    paper = Paper(
        pmid="12345678",
        title="Test paper on aging",
        year=2025,
        abstract="This paper studies cellular senescence.",
    )
    assert paper.pmid == "12345678"
    assert paper.relevance_score == 0.0


def test_target_creation():
    target = Target(
        gene_symbol="TP53",
        gene_name="Tumor protein p53",
        uniprot_id="P04637",
        aging_relevance="Master regulator of cellular senescence",
        mechanism="Activates p21 to induce cell cycle arrest",
        literature_score=0.85,
    )
    assert target.gene_symbol == "TP53"
    assert target.literature_score == 0.85


def test_discovery_report():
    report = DiscoveryReport(
        query="senolytic targets",
        total_papers_analyzed=150,
        targets=[
            Target(gene_symbol="TP53", literature_score=0.9),
            Target(gene_symbol="CDKN2A", literature_score=0.8),
        ],
    )
    assert len(report.targets) == 2
    assert report.total_papers_analyzed == 150
