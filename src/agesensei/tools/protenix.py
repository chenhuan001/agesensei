"""Protenix CLI wrapper for protein structure prediction.

Protenix is ByteDance's open-source AlphaFold3-class model (464M params).
This tool wraps the `protenix pred` CLI to provide async structure prediction.

Install: pip install protenix
Models: protenix_base_default_v1.0.0 (v2, 464M), protenix-mini (135M), protenix-tiny (109M)
"""
from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProtenixResult:
    """Parsed output from a Protenix structure prediction."""
    gene_symbol: str
    sequence: str
    model_used: str
    cif_path: str = ""
    plddt_mean: float = 0.0
    ptm: float = 0.0
    iptm: float = 0.0
    num_residues: int = 0
    prediction_time_sec: float = 0.0
    error: str | None = None
    confidence_data: dict = field(default_factory=dict)


def check_protenix_available() -> bool:
    """Check if protenix CLI is installed and callable."""
    return shutil.which("protenix") is not None


def build_input_json(
    sequences: list[dict],
    output_path: Path | None = None,
) -> dict:
    """Build Protenix input JSON from sequence specifications.

    Args:
        sequences: List of dicts with keys: name, sequence.
            Example: [{"name": "BCL-xL", "sequence": "MSQSNREL..."}]
        output_path: If provided, write JSON to this path.

    Returns:
        The input dict suitable for protenix pred -i.
    """
    entities = []
    for seq in sequences:
        entities.append({
            "type": "protein",
            "sequence": seq["sequence"],
            "count": 1,
        })

    input_data = {
        "name": sequences[0]["name"] if sequences else "prediction",
        "modelSeeds": [1],
        "sequences": entities,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps([input_data], indent=2))

    return input_data


def parse_output(output_dir: Path, gene_symbol: str = "") -> ProtenixResult:
    """Parse Protenix output directory for CIF files and confidence metrics.

    Args:
        output_dir: Directory containing Protenix prediction output.
        gene_symbol: Gene symbol for labeling.

    Returns:
        ProtenixResult with parsed confidence metrics.
    """
    result = ProtenixResult(gene_symbol=gene_symbol, sequence="", model_used="")

    # Find CIF file
    cif_files = list(output_dir.rglob("*.cif"))
    if cif_files:
        result.cif_path = str(cif_files[0])

    # Find confidence JSON (summary_confidence.json or similar)
    confidence_files = list(output_dir.rglob("*confidence*.json"))
    if confidence_files:
        try:
            data = json.loads(confidence_files[0].read_text())
            result.confidence_data = data
            # Extract standard metrics
            if isinstance(data, dict):
                result.ptm = float(data.get("ptm", 0.0))
                result.iptm = float(data.get("iptm", 0.0))
                # pLDDT might be per-residue or averaged
                plddt = data.get("plddt", data.get("mean_plddt", 0.0))
                if isinstance(plddt, list):
                    result.plddt_mean = sum(plddt) / len(plddt) if plddt else 0.0
                    result.num_residues = len(plddt)
                else:
                    result.plddt_mean = float(plddt)
        except (json.JSONDecodeError, ValueError):
            pass

    return result


async def predict_structure(
    sequences: list[dict],
    output_dir: Path | None = None,
    model: str = "protenix_base_default_v1.0.0",
    timeout: int = 600,
) -> ProtenixResult:
    """Run Protenix structure prediction via CLI subprocess.

    Args:
        sequences: List of dicts with keys: name, sequence.
        output_dir: Where to write output. Uses temp dir if None.
        model: Model checkpoint name.
        timeout: Max seconds to wait for prediction.

    Returns:
        ProtenixResult with structure path and confidence metrics.

    Raises:
        RuntimeError: If protenix is not installed or prediction fails.
    """
    if not check_protenix_available():
        return ProtenixResult(
            gene_symbol=sequences[0].get("name", "unknown") if sequences else "unknown",
            sequence=sequences[0].get("sequence", "") if sequences else "",
            model_used=model,
            error="Protenix not installed. Install with: pip install protenix",
        )

    gene_symbol = sequences[0].get("name", "prediction") if sequences else "prediction"
    use_temp = output_dir is None
    if use_temp:
        output_dir = Path(tempfile.mkdtemp(prefix="protenix_"))

    # Write input JSON
    input_json = output_dir / "input.json"
    build_input_json(sequences, input_json)

    # Run protenix pred
    cmd = [
        "protenix", "pred",
        "-i", str(input_json),
        "-o", str(output_dir / "output"),
        "-n", model,
    ]

    import time
    start = time.time()

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )

        elapsed = time.time() - start

        if proc.returncode != 0:
            error_msg = stderr.decode().strip() or f"protenix exited with code {proc.returncode}"
            return ProtenixResult(
                gene_symbol=gene_symbol,
                sequence=sequences[0].get("sequence", "") if sequences else "",
                model_used=model,
                prediction_time_sec=elapsed,
                error=error_msg,
            )

        # Parse output
        result = parse_output(output_dir / "output", gene_symbol)
        result.sequence = sequences[0].get("sequence", "") if sequences else ""
        result.model_used = model
        result.prediction_time_sec = elapsed
        result.num_residues = result.num_residues or len(result.sequence)
        return result

    except asyncio.TimeoutError:
        return ProtenixResult(
            gene_symbol=gene_symbol,
            sequence=sequences[0].get("sequence", "") if sequences else "",
            model_used=model,
            prediction_time_sec=timeout,
            error=f"Prediction timed out after {timeout}s",
        )
    except Exception as e:
        return ProtenixResult(
            gene_symbol=gene_symbol,
            sequence=sequences[0].get("sequence", "") if sequences else "",
            model_used=model,
            error=str(e),
        )
