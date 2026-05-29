"""Global configuration for AgeSensei."""

from pathlib import Path
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
DB_DIR = DATA_DIR / "databases"


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6-20250514"
    max_tokens: int = 4096
    temperature: float = 0.1


class ESMConfig(BaseModel):
    """ESM protein model configuration."""

    model_name: str = "facebook/esm2_t33_650M_UR50D"
    device: str = "cpu"  # "cuda" if GPU available
    batch_size: int = 4


class PubMedConfig(BaseModel):
    """PubMed API configuration."""

    base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    max_results: int = 100
    email: str = ""  # required by NCBI
    api_key: str = ""  # optional, increases rate limit


class SemanticScholarConfig(BaseModel):
    """Semantic Scholar API configuration."""

    base_url: str = "https://api.semanticscholar.org/graph/v1"
    api_key: str = ""
    max_results: int = 100


class AppConfig(BaseModel):
    """Top-level application configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    esm: ESMConfig = Field(default_factory=ESMConfig)
    pubmed: PubMedConfig = Field(default_factory=PubMedConfig)
    semantic_scholar: SemanticScholarConfig = Field(default_factory=SemanticScholarConfig)
    cache_enabled: bool = True
    verbose: bool = True


# Global config instance
config = AppConfig()
