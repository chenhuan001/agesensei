"""Agent modules for AgeSensei pipeline.

Agents are imported lazily to avoid pulling in heavy optional dependencies
(pandas, torch, etc.) at package import time.
"""

__all__ = [
    "LiteratureAgent",
    "TargetExtractor",
    "ProteinAnalyzer",
    "StructurePredictAgent",
    "CADDAgent",
    "DruggabilityAgent",
    "PathwayAgent",
    "BaselineTableAgent",
    "TrendingDigestAgent",
]


def __getattr__(name: str):
    if name == "LiteratureAgent":
        from agesensei.agents.literature import LiteratureAgent
        return LiteratureAgent
    if name == "TargetExtractor":
        from agesensei.agents.target_extractor import TargetExtractor
        return TargetExtractor
    if name == "ProteinAnalyzer":
        from agesensei.agents.protein_analyzer import ProteinAnalyzer
        return ProteinAnalyzer
    if name in ("StructurePredictAgent", "StructurePredictor"):
        from agesensei.agents.structure_predictor import StructurePredictAgent
        return StructurePredictAgent
    if name == "CADDAgent":
        from agesensei.agents.cadd import CADDAgent
        return CADDAgent
    if name == "DruggabilityAgent":
        from agesensei.agents.druggability import DruggabilityAgent
        return DruggabilityAgent
    if name == "PathwayAgent":
        from agesensei.agents.pathway import PathwayAgent
        return PathwayAgent
    if name == "BaselineTableAgent":
        from agesensei.agents.baseline_table import BaselineTableAgent
        return BaselineTableAgent
    if name == "TrendingDigestAgent":
        from agesensei.agents.trending_digest import TrendingDigestAgent
        return TrendingDigestAgent
    raise AttributeError(f"module 'agesensei.agents' has no attribute {name!r}")
