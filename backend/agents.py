"""
Hierarchical Subagent Orchestrator for Text-to-CAD Pipeline.

This module implements the "Chief Engineer" router — a custom Python
orchestrator that makes isolated calls to Gemini 2.5 Flash.
Each function represents a distinct subagent with its own system prompt
and context boundary, preventing spatial hallucinations through strict
context isolation.

Subagent Flow:
  User Prompt → Planner → [Machinist + Critic Loop] × N (concurrent) → Assembler → .glb

The Assembler is now DETERMINISTIC — it loads pre-compiled .step files
and applies manifest translations directly via CadQuery, with zero LLM
involvement. Telemetry is captured for every Machinist attempt.
"""

import asyncio
import json
import pathlib
import re
from typing import Any

import google.generativeai as genai

from classifier import classify_part
from compiler import execute_cad_script
from schemas import AssemblyManifest, PartDefinition
from strategies import get_strategy
from telemetry import log_attempt

MODEL = "gemini-2.5-flash"

# Directory for pre-compiled .step files consumed by the deterministic assembler
_TMP_PARTS_DIR = pathlib.Path(__file__).parent / "tmp_parts"

_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_]")


def _sanitize_part_id(raw: str) -> str:
    """Strip part_id to alphanumeric + underscore only (max 64 chars)."""
    return _SAFE_ID_RE.sub("_", raw)[:64] or "unnamed_part"


def cleanup_tmp_parts() -> None:
    """Remove all files from the tmp_parts directory."""
    if _TMP_PARTS_DIR.exists():
        for f in _TMP_PARTS_DIR.iterdir():
            try:
                f.unlink()
            except OSError:
                pass


class ScriptError(RuntimeError):
    """RuntimeError subclass that carries the last attempted CadQuery script."""

    def __init__(self, message: str, script: str = ""):
        super().__init__(message)
        self.script = script

# Max concurrent Machinist calls (respects Gemini rate limits)
_MACHINIST_SEMAPHORE = asyncio.Semaphore(3)

# Fields that Gemini's structured output schema does NOT support.
_UNSUPPORTED_SCHEMA_KEYS = {"title", "default", "minItems", "maxItems", "minimum", "maximum", "$defs"}


def _sanitize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively strip JSON Schema fields unsupported by the Gemini API."""
    cleaned = {}
    for key, value in schema.items():
        if key in _UNSUPPORTED_SCHEMA_KEYS:
            continue
        if isinstance(value, dict):
            cleaned[key] = _sanitize_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [
                _sanitize_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def _get_gemini_schema() -> dict[str, Any]:
    """Generate a Gemini-compatible JSON schema from AssemblyManifest."""
    raw = AssemblyManifest.model_json_schema()
    defs = raw.pop("$defs", {})
    raw_json = json.dumps(raw)
    for def_name, def_schema in defs.items():
        ref_str = f'{{"$ref": "#/$defs/{def_name}"}}'
        raw_json = raw_json.replace(ref_str, json.dumps(def_schema))
    resolved = json.loads(raw_json)
    return _sanitize_schema(resolved)


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    text = re.sub(r"^```(?:python|py)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


async def run_planner(user_prompt: str) -> AssemblyManifest:
    """Subagent 1: The Draftsman.

    Converts a natural-language mechanical request into a structured
    AssemblyManifest using Gemini's structured output mode.
    """
    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=(
            "You are the Chief Draftsman for a mechanical CAD system. "
            "Convert the user's mechanical request into a strict AssemblyManifest JSON. "
            "Follow ALL anchor_tag and mating_rule constraints defined in the schema. "
            "RULES: "
            "(1) Every part must be self-contained and modeled at the origin. "
            "(2) anchor_tags MUST be standard CadQuery directional selectors ONLY: "
            "'>Z' (top), '<Z' (bottom), '>X' (right), '<X' (left), '>Y' (front), '<Y' (back). "
            "NEVER use custom named anchors like 'face_top_horizontal' or 'hole_center'. "
            "NEVER use coordinate tuples. "
            "(3) mating_rules must reference only existing part_ids and anchor_tags. "
            "(4) The assembly graph must be fully connected — no orphan parts. "
            "(5) List parts in logical build order: base/foundation first, fasteners last. "
            "(6) All dimensions must be in millimeters. "
            "\n"
            "SINGLE-PART vs MULTI-PART DETECTION: "
            "If the user describes a SINGLE object (e.g., 'a gear', 'a bolt', 'a bracket'), "
            "create EXACTLY ONE PartDefinition. For a single-part assembly, create one "
            "mating_rule where source and target are the SAME part_id (self-reference), "
            "with translation '0, 0, 0' and clearance 0.0. This signals the pipeline "
            "to skip the assembly step and export the part directly. "
            "\n"
            "DOMAIN D AEROSPACE SUBSYSTEM RULE (OVERRIDES MULTI-PART RULE A): "
            "When the request involves an aerodynamic structure with internal "
            "reinforcement — specifically a wing, stabilizer, fin, or airfoil "
            "segment that contains ribs, spars, bulkheads, or any internal "
            "structural members — you MUST NOT decompose it into separate "
            "PartDefinition entries (e.g., do NOT create 'wing_skin', 'rib_1', "
            "'rib_2', 'spar_front' as separate parts). "
            "Instead, define a SINGLE PartDefinition with part_id "
            "'wing_segment_assembly' (or an appropriate name). In the "
            "description field, explicitly instruct the Machinist to: "
            "  (1) Loft the airfoil core solid — this IS the inner_void. "
            "  (2) Shell the inner_void OUTWARD (positive wall thickness) to grow the skin. "
            "  (3) Generate all ribs and spars by intersecting oversized blanks "
            "      with inner_void. "
            "  (4) Package everything into a cq.Assembly and assign it to result. "
            "Include the requested rib count, spar count, NACA code, chord, span, "
            "and any other parameters in the description. "
            "Use anchor_tags: ['>Z', '<Z', '>X', '<X', '<Y', '>Y']. "
            "This is MANDATORY because internal ribs depend on a shared boolean "
            "tooling body (the inner_void) that cannot be split across scripts. "
            "Create a self-referencing mating_rule with translation '0, 0, 0'. "
            "\n"
            "MULTI-PART PLANNING RULES: "
            "(A) For NON-aerospace assemblies, you MUST break down every multi-part "
            "assembly into DISTINCT, INDIVIDUALLY NAMED parts. Do NOT group identical "
            "parts into a single entry. For example, a table must have 'tabletop', "
            "'leg_1', 'leg_2', 'leg_3', 'leg_4' — five separate PartDefinition "
            "entries, NOT 'tabletop' and 'legs'. Even if parts are geometrically "
            "identical, each instance must be its own entry with a unique part_id. "
            "(B) Each MatingRule's 'translation' field MUST contain the precise absolute "
            "3D coordinate (X, Y, Z) in millimeters for where to place the target part. "
            "The base/first part is always at '0, 0, 0'."
        ),
    )

    response = await model.generate_content_async(
        user_prompt,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": _get_gemini_schema(),
        },
    )

    return AssemblyManifest.model_validate_json(response.text)


def is_single_part(manifest: AssemblyManifest) -> bool:
    """Check if the manifest describes a single part (no real assembly needed)."""
    if len(manifest.parts) == 1:
        return True
    return False


async def run_machinist(
    part_def: PartDefinition, domain: str = "A", error_context: str = "",
) -> str:
    """Subagent 2: The Machinist.

    Generates isolated CadQuery Python code for a single part.
    The domain must be pre-classified by the caller; the system prompt
    is built dynamically based on domain and part description keywords.
    """
    system_prompt = get_strategy(domain, part_def.description)

    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=system_prompt,
    )

    prompt = (
        f"Generate CadQuery code for this part:\n"
        f"Part ID: {part_def.part_id}\n"
        f"Description: {part_def.description}\n"
        f"Anchor Tags: {part_def.anchor_tags}\n"
    )

    if error_context:
        prompt += (
            f"\nYour previous attempt FAILED with this error:\n"
            f"---\n{error_context}\n---\n"
            f"Fix the CadQuery code to resolve this error. "
            f"Return the complete corrected script."
        )

    response = await model.generate_content_async(prompt)
    return _strip_markdown_fences(response.text)


async def _save_part_step(part_id: str, script: str) -> str:
    """Save a compiled part as an isolated .step file for the deterministic assembler.

    Re-executes the validated script with an appended export stanza.
    Returns the absolute path to the saved .step file.
    """
    safe_id = _sanitize_part_id(part_id)
    _TMP_PARTS_DIR.mkdir(exist_ok=True)
    step_path = (_TMP_PARTS_DIR / f"{safe_id}.step").resolve()
    if not str(step_path).startswith(str(_TMP_PARTS_DIR.resolve())):
        raise ValueError(f"Path traversal detected in part_id: {part_id!r}")
    safe_path = step_path.as_posix()

    export_script = (
        f"{script}\n\n"
        f"import cadquery as cq\n"
        f"if isinstance(result, cq.Assembly):\n"
        f"    result.save(r'{safe_path}')\n"
        f"else:\n"
        f"    cq.exporters.export(result, r'{safe_path}')\n"
    )
    await execute_cad_script(export_script)
    return safe_path


async def run_critic_loop(
    part_def: PartDefinition, max_retries: int = 3,
) -> tuple[str, str]:
    """Subagent 3: The QA Inspector (Critic Loop).

    Runs the Machinist's code via subprocess with concurrency control.
    On failure, feeds the traceback back for self-correction.

    Returns:
        (validated_code, step_file_path) — the clean CadQuery source and
        the absolute path to the pre-compiled .step file.
    """
    async with _MACHINIST_SEMAPHORE:
        # Classify once per part — domain is stable across retries
        classification = await classify_part(part_def.description)
        domain = classification.get("domain", "A")

        error_context = ""

        for attempt in range(1, max_retries + 1):
            code = await run_machinist(part_def, domain=domain, error_context=error_context)
            result = await execute_cad_script(code)

            success = result["status"] == "success"
            log_attempt(
                part_id=part_def.part_id,
                domain=domain,
                attempt=attempt,
                code=code,
                error=None if success else result.get("traceback"),
                success=success,
            )

            if success:
                step_path = await _save_part_step(part_def.part_id, code)
                return code, step_path

            error_context = result["traceback"]
            if attempt == max_retries:
                raise ScriptError(
                    f"Part '{part_def.part_id}' failed after {max_retries} attempts. "
                    f"Last error:\n{error_context}",
                    script=code,
                )

        raise ScriptError(f"Part '{part_def.part_id}' failed unexpectedly.")


async def run_machinist_batch(
    parts: list[PartDefinition],
) -> tuple[dict[str, str], dict[str, str]]:
    """Run all Machinist critic loops concurrently with rate limiting.

    Deduplicates parts with identical descriptions: calls the Machinist
    only once per unique description and reuses the cached code for all
    identical parts.

    Returns:
        (part_scripts, step_files) — two dicts mapping part_id to
        validated code and to the pre-compiled .step file path.
    """
    # Group parts by description for deduplication
    desc_to_parts: dict[str, list[PartDefinition]] = {}
    for part in parts:
        desc_to_parts.setdefault(part.description, []).append(part)

    # Build one representative per unique description
    unique_parts = [group[0] for group in desc_to_parts.values()]

    tasks = [run_critic_loop(part) for part in unique_parts]
    results = await asyncio.gather(*tasks)

    # Map representative description -> (code, step_path)
    desc_to_result = {
        part.description: result for part, result in zip(unique_parts, results)
    }

    # Fan out cached results to all parts sharing the same description
    part_scripts: dict[str, str] = {}
    step_files: dict[str, str] = {}
    for part in parts:
        code, step_path = desc_to_result[part.description]
        part_scripts[part.part_id] = code
        step_files[part.part_id] = step_path

    return part_scripts, step_files


async def run_single_part_export(part_script: str, output_filename: str) -> str:
    """Generate a simple export script for single-part models (no assembly).

    If the Machinist already produced a cq.Assembly (e.g., Domain D subsystem),
    we save it directly instead of nesting it inside another assembly.
    """
    return (
        f"{part_script}\n\n"
        f"import cadquery as cq\n"
        f"if isinstance(result, cq.Assembly):\n"
        f"    result.save('{output_filename}')\n"
        f"else:\n"
        f"    _export_asm = cq.Assembly()\n"
        f"    _export_asm.add(result, name='part')\n"
        f"    _export_asm.save('{output_filename}')\n"
    )


# --------------------------------------------------------------------------
# Phase 1: Deterministic Assembler (ZERO LLM involvement)
# --------------------------------------------------------------------------

def run_assembler(
    manifest: AssemblyManifest,
    step_files: dict[str, str],
) -> str:
    """Deterministic assembler — generates a CadQuery script from .step files.

    Loads each pre-compiled .step file via cq.importers.importStep(),
    applies the exact X, Y, Z translations from the manifest's mating_rules,
    adds them to a cq.Assembly, and saves the final .glb and .step files.

    No LLM is involved. The output is a fully deterministic script.
    """
    lines = ["import cadquery as cq", ""]
    lines.append("assembly = cq.Assembly()")
    lines.append("")

    added: set[str] = set()

    # Add each part at its manifest-specified position
    for part in manifest.parts:
        pid = part.part_id
        if pid in added or pid not in step_files:
            continue

        safe_pid = _sanitize_part_id(pid)

        # Find translation: look for a mating rule where this part is the target.
        # The base/first part is never a target — defaults to origin (0, 0, 0).
        tx, ty, tz = 0.0, 0.0, 0.0
        for rule in manifest.mating_rules:
            if rule.target_part_id == pid:
                coords = [float(v.strip()) for v in rule.translation.split(",")]
                tx, ty, tz = coords[0], coords[1], coords[2]
                break

        safe_path = step_files[pid].replace("\\", "/")
        var = safe_pid if safe_pid.isidentifier() else f"part_{safe_pid}"

        lines.append(f"{var} = cq.importers.importStep(r'{safe_path}')")
        lines.append(
            f"assembly.add({var}, name='{safe_pid}', "
            f"loc=cq.Location(cq.Vector({tx}, {ty}, {tz})))"
        )
        lines.append("")
        added.add(pid)

    lines.append("assembly.save('output.glb')")
    lines.append("assembly.save('output.step')")

    return "\n".join(lines)


async def run_assembly_critic_loop(
    manifest: AssemblyManifest,
    step_files: dict[str, str],
) -> str:
    """Execute the deterministic assembler and return the script.

    Unlike the old LLM-based assembler, this has no retry loop — the
    script is deterministic. If it fails, the error is structural
    (bad .step file or missing part) and retrying won't help.
    """
    if len(step_files) != len(manifest.parts):
        raise ScriptError(
            f"Assembly failed: Missing compiled parts. Expected "
            f"{len(manifest.parts)} parts, but only got {len(step_files)}."
        )

    script = run_assembler(manifest, step_files)
    result = await execute_cad_script(script)

    if result["status"] != "success":
        raise ScriptError(
            f"Deterministic assembly failed:\n{result['traceback']}",
            script=script,
        )

    return script
