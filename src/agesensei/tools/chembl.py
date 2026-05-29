"""ChEMBL database API wrapper for bioactivity and drug data.

ChEMBL: https://www.ebi.ac.uk/chembl/
API: https://chembl.gitbook.io/chembl-interface-documentation/web-services
Free, no API key needed.
"""

import asyncio
import httpx

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
MAX_RETRIES = 3


async def _get(client: httpx.AsyncClient, url: str, params: dict) -> dict:
    """GET with retry."""
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.get(url, params=params)
            if response.status_code == 429:
                await asyncio.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            return response.json()
        except (httpx.ConnectError, httpx.ReadTimeout):
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                raise
    return {}


async def get_target_by_gene(gene_symbol: str, organism: str = "Homo sapiens") -> dict | None:
    """Find ChEMBL target entry for a gene symbol.

    Returns:
        Dict with target_chembl_id, pref_name, target_type, organism
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        data = await _get(client, f"{CHEMBL_API}/target/search.json", {
            "q": gene_symbol,
            "limit": 5,
        })

    for target in data.get("targets", []):
        # Match by gene name and organism
        if (target.get("organism", "") == organism and
                target.get("target_type", "") == "SINGLE PROTEIN"):
            # Verify gene symbol matches
            components = target.get("target_components", [])
            for comp in components:
                synonyms = comp.get("target_component_synonyms", [])
                gene_names = [s["component_synonym"] for s in synonyms
                              if s.get("syn_type") == "GENE_SYMBOL"]
                if gene_symbol.upper() in [g.upper() for g in gene_names]:
                    return {
                        "target_chembl_id": target["target_chembl_id"],
                        "pref_name": target.get("pref_name", ""),
                        "target_type": target.get("target_type", ""),
                        "organism": target.get("organism", ""),
                    }

    # Fallback: return first human single protein match
    for target in data.get("targets", []):
        if (target.get("organism", "") == organism and
                target.get("target_type", "") == "SINGLE PROTEIN"):
            return {
                "target_chembl_id": target["target_chembl_id"],
                "pref_name": target.get("pref_name", ""),
                "target_type": target.get("target_type", ""),
                "organism": target.get("organism", ""),
            }

    return None


async def get_bioactivities(target_chembl_id: str, max_results: int = 50) -> list[dict]:
    """Get bioactivity data (IC50, Ki, EC50, etc.) for a target.

    Returns list of dicts: molecule_chembl_id, molecule_name, type, value, units
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        data = await _get(client, f"{CHEMBL_API}/activity.json", {
            "target_chembl_id": target_chembl_id,
            "limit": max_results,
            "standard_type__in": "IC50,Ki,EC50,Kd",
        })

    activities = []
    for act in data.get("activities", []):
        activities.append({
            "molecule_chembl_id": act.get("molecule_chembl_id", ""),
            "molecule_name": act.get("canonical_smiles", "")[:50],
            "activity_type": act.get("standard_type", ""),
            "value": act.get("standard_value"),
            "units": act.get("standard_units", ""),
            "pchembl_value": act.get("pchembl_value"),
        })
    return activities


async def get_approved_drugs(target_chembl_id: str) -> list[dict]:
    """Get approved drugs targeting this protein."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        data = await _get(client, f"{CHEMBL_API}/mechanism.json", {
            "target_chembl_id": target_chembl_id,
            "limit": 50,
        })

    drugs = []
    seen = set()
    for mech in data.get("mechanisms", []):
        mol_id = mech.get("molecule_chembl_id", "")
        if mol_id in seen:
            continue
        seen.add(mol_id)

        # Fetch molecule details
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                mol_data = await _get(client, f"{CHEMBL_API}/molecule/{mol_id}.json", {})
            max_phase = mol_data.get("max_phase", 0)
            pref_name = mol_data.get("pref_name", "")
        except Exception:
            max_phase = 0
            pref_name = ""

        drugs.append({
            "molecule_chembl_id": mol_id,
            "name": pref_name or mol_id,
            "mechanism": mech.get("mechanism_of_action", ""),
            "max_phase": max_phase,  # 4 = approved
            "action_type": mech.get("action_type", ""),
        })
    return drugs


async def assess_druggability(gene_symbol: str) -> dict:
    """Quick druggability assessment for a gene.

    Returns dict with: target_chembl_id, has_bioactivities, n_bioactivities,
    has_approved_drugs, approved_drugs, max_phase
    """
    target = await get_target_by_gene(gene_symbol)
    if not target:
        return {
            "gene": gene_symbol,
            "found_in_chembl": False,
            "has_bioactivities": False,
            "n_bioactivities": 0,
            "has_approved_drugs": False,
            "approved_drugs": [],
            "max_phase": 0,
        }

    chembl_id = target["target_chembl_id"]

    # Fetch bioactivities and drugs in parallel
    activities, drugs = await asyncio.gather(
        get_bioactivities(chembl_id, max_results=20),
        get_approved_drugs(chembl_id),
    )

    approved = [d for d in drugs if d.get("max_phase", 0) >= 4]

    return {
        "gene": gene_symbol,
        "found_in_chembl": True,
        "target_chembl_id": chembl_id,
        "target_name": target["pref_name"],
        "has_bioactivities": len(activities) > 0,
        "n_bioactivities": len(activities),
        "has_approved_drugs": len(approved) > 0,
        "approved_drugs": [d["name"] for d in approved],
        "max_phase": max(d.get("max_phase", 0) for d in drugs) if drugs else 0,
    }
