"""Target extraction agent: identifies gene/protein targets from literature.

Uses LLM-based NER/RE or regex fallback to extract:
1. Gene/protein names mentioned in abstracts
2. Gene-disease-function relationship triples
3. Cross-validates against GenAge/CellAge databases

Two modes:
- LLM mode: high-quality extraction using Claude
- Regex mode: basic gene name extraction using pattern matching (fallback)
"""

import re
from agesensei.schema import Paper, Target
from agesensei.tools.hagr import is_known_aging_gene, get_aging_evidence, get_all_aging_genes

# Try to import anthropic
try:
    import anthropic
    import os
    HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))
except ImportError:
    HAS_ANTHROPIC = False

# Common human gene name pattern
GENE_PATTERN = re.compile(r'\b([A-Z][A-Z0-9]{1,8}(?:-[A-Z0-9]+)?)\b')

# Words that look like gene names but aren't
FALSE_POSITIVES = {
    "DNA", "RNA", "ATP", "ADP", "GTP", "NAD", "NADH", "FAD", "NMN",
    "PCR", "ROS", "RCT", "FDA", "USA", "EU", "WHO", "BMI", "MRI",
    "DMSO", "PBS", "EDTA", "BSA", "ANOVA", "SEM", "SD",
    "NOT", "AND", "THE", "FOR", "ARE", "WAS", "HAS", "HAD",
    "ALL", "ANY", "BUT", "CAN", "DID", "GET", "HER", "HIS",
    "LET", "MAY", "NEW", "NOW", "OLD", "OUR", "OUT", "OWN",
    "ITS", "HOW", "MAN", "DAY", "AGE", "END", "TWO",
    "MICE", "CELL", "CELLS", "GENE", "GENES", "DRUG", "DOSE",
    "MALE", "HUMAN", "MOUSE", "MODEL", "STUDY", "DATA",
    "AGING", "LIFE", "DEATH", "RISK", "ROLE", "TYPE", "LOSS",
    "HIGH", "LONG", "TERM", "TIME", "YEAR", "WEEK",
    "RESULTS", "METHODS", "CONCLUSION", "BACKGROUND",
    "PROTEIN", "PATHWAY", "RECEPTOR", "INHIBITOR", "TREATMENT",
    "CANCER", "TUMOR", "TRIAL", "PHASE", "GROUP", "LEVEL",
    "COMPARED", "INCREASED", "DECREASED", "INDUCED", "REDUCED",
    "CONTROL", "EFFECT", "EXPRESSION", "ANALYSIS", "FUNCTION",
    "ASSOCIATED", "RELATED", "SPECIFIC", "SIGNIFICANT",
    "MECHANISM", "PROCESS", "DISEASE", "ACTIVITY", "FACTOR",
    "BASED", "USING", "THESE", "THOSE", "THAN", "THAT",
    "ALSO", "BOTH", "EACH", "FROM", "HAVE", "INTO", "MORE",
    "ONLY", "OVER", "SOME", "SUCH", "THEY", "VERY", "WHEN",
    "WILL", "WITH", "BEEN", "WERE", "MOST", "MANY",
    "SENOLYTICS", "SENOLYTIC", "SENESCENCE",
    # Common abbreviations that aren't genes
    "AI", "ML", "CNS", "PNS", "BBB", "GI", "IV", "IM", "SC",
    "ICU", "ER", "OR", "HR", "CI", "NS", "NF", "DDR",
    "COPD", "CVD", "CHF", "CKD", "IBD", "IBS",
    "SASP", "DAMP", "PAMP", "TLR",  # biology terms not gene names
    "OAG", "DME", "IPF", "ALS", "NAION",  # disease abbreviations
    "AAV", "LNP", "MOI", "GFP", "RFP",  # lab/tool terms
    "TRYP", "ABT263",  # drug names not gene names
    "KPA", "DEXA", "CGM", "HRV",  # medical device/test terms
}

EXTRACTION_PROMPT = """Extract all gene and protein targets mentioned in this abstract. For each target, identify:
1. The gene/protein symbol (standardized human gene nomenclature, e.g., TP53 not p53)
2. Its role/function described in the paper
3. Whether it is proposed as a drug target
4. The relationship to aging/senescence if mentioned

Abstract:
{abstract}

Respond in this exact format (one target per line):
GENE: <symbol> | ROLE: <brief role> | DRUGGABLE: <yes/no/unclear> | AGING_LINK: <description or none>

If no genes/proteins are mentioned, respond with: NO_TARGETS_FOUND"""


class TargetExtractor:
    """Extract gene/protein targets from paper abstracts.

    Pipeline:
        1. Extract gene names from abstract (LLM or regex)
        2. Standardize names
        3. Cross-reference with GenAge/CellAge
        4. Score target relevance

    Example:
        extractor = TargetExtractor()
        targets = extractor.extract_from_papers(papers)
    """

    def __init__(self):
        self.client = None
        self.use_llm = False

        if HAS_ANTHROPIC:
            self.client = anthropic.Anthropic()
            self.use_llm = True

        self._aging_genes = None

    @property
    def aging_genes(self) -> set[str]:
        if self._aging_genes is None:
            self._aging_genes = get_all_aging_genes()
        return self._aging_genes

    def extract_from_papers(self, papers: list[Paper]) -> list[Target]:
        """Extract targets from a list of papers.

        Args:
            papers: List of Paper objects (must have abstract)

        Returns:
            List of Target objects, deduplicated and scored
        """
        all_targets: dict[str, Target] = {}

        mode = "LLM" if self.use_llm else "regex"
        print(f"  Extracting targets from {len(papers)} papers (mode: {mode})...")

        for i, paper in enumerate(papers):
            if not paper.abstract:
                continue

            if self.use_llm:
                raw_targets = self._extract_with_llm(paper)
            else:
                raw_targets = self._extract_with_regex(paper)

            # Merge into aggregated target list
            for t in raw_targets:
                symbol = t.gene_symbol.upper()
                if symbol in all_targets:
                    existing = all_targets[symbol]
                    existing.paper_count += 1
                    existing.source_papers.append(paper.pmid or paper.doi or "")
                    if t.role and t.role not in (existing.role or ""):
                        existing.role = f"{existing.role}; {t.role}" if existing.role else t.role
                else:
                    all_targets[symbol] = t

        # Enrich with HAGR data and score
        targets = list(all_targets.values())
        for target in targets:
            self._enrich_with_hagr(target)
            self._score_target(target)

        # Sort by score
        targets.sort(key=lambda t: t.score, reverse=True)
        print(f"  Extracted {len(targets)} unique targets")
        return targets

    def _extract_with_regex(self, paper: Paper) -> list[Target]:
        """Extract gene names using regex pattern matching."""
        text = f"{paper.title} {paper.abstract}"
        candidates = GENE_PATTERN.findall(text)

        # Also look for common gene name patterns with hyphens/numbers
        # e.g., "p53", "p21", "BCL-2", "BCL-xL", "IL-6"
        lowercase_genes = re.findall(r'\b(p\d{1,3}|bcl-?\w+|il-?\d+|tnf-?\w*|igf-?\d*|tgf-?\w*)\b', text, re.IGNORECASE)
        candidates.extend([g.upper().replace("-", "") for g in lowercase_genes])

        targets = []
        seen = set()
        for candidate in candidates:
            candidate = self._normalize_gene_name(candidate)
            if candidate in seen or candidate in FALSE_POSITIVES or len(candidate) < 2:
                continue

            # Extra validation: skip if it's just a number
            if candidate.isdigit():
                continue

            # Bonus: prioritize if it's a known aging gene
            seen.add(candidate)
            target = Target(
                gene_symbol=candidate,
                paper_count=1,
                source_papers=[paper.pmid or paper.doi or ""],
            )
            targets.append(target)

        return targets

    def _normalize_gene_name(self, name: str) -> str:
        """Normalize gene name to standard symbol."""
        name = name.upper().strip()

        # Common aliases
        aliases = {
            "P53": "TP53", "P21": "CDKN1A", "P16": "CDKN2A",
            "P27": "CDKN1B", "P14ARF": "CDKN2A", "RB": "RB1",
            "BCL2": "BCL2", "BCLXL": "BCL2L1", "BCL-XL": "BCL2L1",
            "BCL-2": "BCL2", "BCLX": "BCL2L1",
            "IL6": "IL6", "IL1B": "IL1B", "IL-6": "IL6", "IL-1": "IL1B",
            "TNFA": "TNF", "TNF-A": "TNF", "TNFALPHA": "TNF",
            "TGFB": "TGFB1", "TGF-B": "TGFB1",
            "IGFI": "IGF1", "IGF-1": "IGF1",
        }
        return aliases.get(name, name)

    def _extract_with_llm(self, paper: Paper) -> list[Target]:
        """Extract targets using LLM-based NER/RE."""
        prompt = EXTRACTION_PROMPT.format(abstract=paper.abstract[:2000])

        try:
            response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1000,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text.strip()
            if "NO_TARGETS_FOUND" in text:
                return []

            targets = []
            for line in text.split("\n"):
                line = line.strip()
                if not line.startswith("GENE:"):
                    continue
                target = self._parse_llm_line(line, paper)
                if target:
                    targets.append(target)
            return targets

        except Exception:
            return self._extract_with_regex(paper)

    def _parse_llm_line(self, line: str, paper: Paper) -> Target | None:
        """Parse a single LLM extraction line."""
        try:
            parts = line.split("|")
            gene_symbol = parts[0].replace("GENE:", "").strip().upper()
            role = parts[1].replace("ROLE:", "").strip() if len(parts) > 1 else ""
            druggable_str = parts[2].replace("DRUGGABLE:", "").strip().lower() if len(parts) > 2 else "unclear"
            aging_link = parts[3].replace("AGING_LINK:", "").strip() if len(parts) > 3 else ""

            is_druggable = None
            if druggable_str == "yes":
                is_druggable = True
            elif druggable_str == "no":
                is_druggable = False

            return Target(
                gene_symbol=gene_symbol,
                role=role,
                is_druggable=is_druggable,
                aging_link=aging_link if aging_link.lower() != "none" else "",
                paper_count=1,
                source_papers=[paper.pmid or paper.doi or ""],
            )
        except Exception:
            return None

    def _enrich_with_hagr(self, target: Target):
        """Enrich target with GenAge/CellAge data."""
        evidence = get_aging_evidence(target.gene_symbol)
        if evidence:
            target.in_genage = 'genage' in evidence
            target.in_cellage = 'cellage' in evidence

            if 'genage' in evidence:
                target.gene_name = target.gene_name or evidence['genage'].get('name', '')

            if 'cellage' in evidence:
                effect = evidence['cellage'].get('senescence_effect', '')
                if effect and not target.aging_link:
                    target.aging_link = f"CellAge: {effect} senescence"

    def _score_target(self, target: Target):
        """Score target relevance based on multiple factors."""
        score = 0.0
        score += min(target.paper_count * 0.15, 0.3)  # paper count
        if target.in_genage:
            score += 0.25
        if target.in_cellage:
            score += 0.15
        if target.aging_link:
            score += 0.1
        if target.is_druggable is True:
            score += 0.2
        target.score = min(score, 1.0)
