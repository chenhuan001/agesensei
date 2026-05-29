"""Druggability assessment agent: evaluates how targetable a protein is.

Integrates:
    - ChEMBL: bioactivity data, known drugs, mechanism of action
    - Open Targets: tractability, safety, disease associations, known drugs
"""

import asyncio
from agesensei.schema import Target, DrugabilityAssessment, DrugInfo
from agesensei.tools import chembl, opentargets


class DruggabilityAgent:
    """Assess the druggability of target proteins.

    Combines ChEMBL (bioactivity data) and Open Targets (tractability, safety,
    disease links) to produce a composite druggability score.

    Example:
        agent = DruggabilityAgent()
        result = await agent.assess_gene("BCL2")
    """

    async def assess_gene(self, gene_symbol: str) -> DrugabilityAssessment:
        """Full druggability assessment for a gene symbol."""
        print(f"  Assessing druggability: {gene_symbol}...")

        # Query both sources, but don't fail if one is down
        chembl_result, ot_result = await asyncio.gather(
            self._safe_call(chembl.assess_druggability, gene_symbol),
            self._safe_call(opentargets.assess_target, gene_symbol),
        )

        # Parse known drugs from both sources
        known_drugs = self._merge_drugs(chembl_result, ot_result)

        # Parse tractability
        tractability = ""
        if ot_result and ot_result.get("tractability"):
            modalities = []
            for modality, labels in ot_result["tractability"].items():
                modalities.append(f"{modality}: {', '.join(labels[:2])}")
            tractability = "; ".join(modalities)

        # Compute score
        score = self._compute_score(chembl_result, ot_result, known_drugs)

        # Build result
        has_approved = any(d.status == "approved" for d in known_drugs)

        result = DrugabilityAssessment(
            gene_symbol=gene_symbol,
            has_known_drugs=len(known_drugs) > 0,
            known_drugs=known_drugs,
            protein_class="",  # would need additional lookup
            tractability=tractability,
            score=score,
            reasoning=self._generate_reasoning(gene_symbol, chembl_result, ot_result, known_drugs),
        )

        status = "druggable" if score > 0.6 else "moderate" if score > 0.3 else "challenging"
        print(f"    {gene_symbol}: score={score:.2f} ({status}), {len(known_drugs)} known drugs")
        return result

    async def assess_targets(self, targets: list[Target], top_n: int = 5) -> dict[str, DrugabilityAssessment]:
        """Assess druggability for top N targets."""
        print(f"\nAssessing druggability for top {min(top_n, len(targets))} targets...")
        results = {}
        for target in targets[:top_n]:
            try:
                assessment = await self.assess_gene(target.gene_symbol)
                results[target.gene_symbol] = assessment
            except Exception as e:
                print(f"    {target.gene_symbol}: failed - {e}")
        return results

    async def _safe_call(self, func, *args):
        """Call async function, return empty dict on failure."""
        try:
            return await func(*args)
        except Exception:
            return {}

    def _merge_drugs(self, chembl_result: dict, ot_result: dict) -> list[DrugInfo]:
        """Merge drug info from ChEMBL and Open Targets."""
        drugs = []
        seen = set()

        # From Open Targets (richer data)
        if ot_result:
            for d in ot_result.get("known_drugs", []):
                name = d.get("name", "")
                if name and name not in seen:
                    seen.add(name)
                    phase = d.get("max_phase", 0)
                    drugs.append(DrugInfo(
                        drug_name=name,
                        mechanism=d.get("mechanism", ""),
                        status="approved" if phase >= 4 else f"phase {phase}",
                        indication=d.get("indication", ""),
                    ))

        # From ChEMBL (add any not already found)
        if chembl_result:
            for name in chembl_result.get("approved_drugs", []):
                if name and name not in seen:
                    seen.add(name)
                    drugs.append(DrugInfo(
                        drug_name=name,
                        status="approved",
                    ))

        return drugs

    def _compute_score(self, chembl_result: dict, ot_result: dict, drugs: list[DrugInfo]) -> float:
        """Compute composite druggability score (0-1)."""
        score = 0.0

        # Has approved drugs (strong signal)
        if any(d.status == "approved" for d in drugs):
            score += 0.3

        # Has clinical-stage drugs
        if drugs:
            score += 0.15

        # ChEMBL bioactivity data exists
        if chembl_result and chembl_result.get("has_bioactivities"):
            n_act = chembl_result.get("n_bioactivities", 0)
            score += min(0.2, n_act * 0.02)  # up to 0.2 for 10+ activities

        # Open Targets tractability
        if ot_result and ot_result.get("tractability"):
            n_modalities = len(ot_result["tractability"])
            score += min(0.2, n_modalities * 0.1)

        # Found in ChEMBL at all
        if chembl_result and chembl_result.get("found_in_chembl"):
            score += 0.1

        # Safety: fewer signals = better
        if ot_result:
            safety = ot_result.get("safety_signals", 0)
            if safety == 0:
                score += 0.05

        return min(score, 1.0)

    def _generate_reasoning(self, gene: str, chembl_result: dict, ot_result: dict, drugs: list[DrugInfo]) -> str:
        """Generate human-readable druggability reasoning."""
        parts = []

        if drugs:
            approved = [d.drug_name for d in drugs if d.status == "approved"]
            if approved:
                parts.append(f"Approved drugs: {', '.join(approved[:3])}")
            clinical = [d for d in drugs if d.status != "approved"]
            if clinical:
                parts.append(f"{len(clinical)} drugs in clinical development")
        else:
            parts.append("No known drugs")

        if chembl_result and chembl_result.get("has_bioactivities"):
            parts.append(f"{chembl_result['n_bioactivities']} bioactivity records in ChEMBL")

        if ot_result and ot_result.get("tractability"):
            modalities = list(ot_result["tractability"].keys())
            parts.append(f"Tractable by: {', '.join(modalities)}")

        if ot_result and ot_result.get("top_diseases"):
            top_disease = ot_result["top_diseases"][0]["disease_name"]
            parts.append(f"Top disease association: {top_disease}")

        return "; ".join(parts) if parts else "Insufficient data"
