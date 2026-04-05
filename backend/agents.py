"""
Hierarchical Subagent Orchestrator for Text-to-CAD Pipeline.

This module implements the "Chief Engineer" router — a custom Python
orchestrator that makes isolated calls to Gemini 2.5 Flash.
Each function represents a distinct subagent with its own system prompt
and context boundary, preventing spatial hallucinations through strict
context isolation.

Subagent Flow:
  User Prompt → Planner → [Machinist + Critic Loop] × N (concurrent) → Assembler → .glb
"""

import asyncio
import json
import re
from typing import Any

import google.generativeai as genai

from compiler import execute_cad_script
from schemas import AssemblyManifest, PartDefinition

MODEL = "gemini-2.5-flash"


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
            "MULTI-PART PLANNING RULES: "
            "(A) You MUST break down every multi-part assembly into DISTINCT, INDIVIDUALLY "
            "NAMED parts. Do NOT group identical parts into a single entry. For example, a "
            "table must have 'tabletop', 'leg_1', 'leg_2', 'leg_3', 'leg_4' — five "
            "separate PartDefinition entries, NOT 'tabletop' and 'legs'. Even if parts "
            "are geometrically identical, each instance must be its own entry with a "
            "unique part_id. "
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


async def run_machinist(part_def: PartDefinition, error_context: str = "") -> str:
    """Subagent 2: The Machinist.

    Generates isolated CadQuery Python code for a single part.
    Receives ONLY the part definition — no assembly-level context.
    """
    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=(
            "You are an expert CadQuery machinist. Your sole job is to write "
            "isolated, raw Python code that generates a single mechanical part "
            "using the CadQuery library. "
            "RULES: "
            "(1) Import cadquery as cq at the top of the script. "
            "(2) The script must define a variable named 'result' containing "
            "the final cq.Workplane object. "
            "(3) Model the part centered at the origin. "
            "(4) Use ONLY the dimensions and features described in the part definition. "
            "(5) Do NOT include any markdown formatting, code fences, or comments "
            "explaining what the code does. Return ONLY raw executable Python. "
            "(6) Do NOT import or reference any other parts or assemblies "
            "(EXCEPT cad_utils — see GEAR RULE below). "
            "\n"
            "MANDATORY 3D SOLID RULE: "
            "You MUST ALWAYS create 3D solids. After drawing any 2D sketch (like "
            ".rect(), .circle(), .polyline()), you MUST explicitly call .extrude() "
            "with a non-zero depth to produce a solid body. NEVER return a flat 2D "
            "sketch, a 0-thickness part, or an unextruded cq.Workplane. Every 'result' "
            "variable must contain a 3D solid with real volume. "
            "\n"
            "GEAR RULE (MANDATORY — USE THE LIBRARY): "
            "For ANY gear (spur gear, pinion, etc.), you MUST use the pre-built "
            "utility function. Do NOT write gear math yourself. Use EXACTLY this: "
            "  from cad_utils import make_involute_spur_gear "
            "  result = make_involute_spur_gear( "
            "      num_teeth=20, "
            "      module=2.0, "
            "      pressure_angle_deg=20.0, "
            "      thickness=10.0, "
            "      bore_diameter=8.0, "
            "      pitch_diameter=None "
            "  ) "
            "Parameters: "
            "- num_teeth: number of teeth. "
            "- module: gear module in mm (standard: module = pitch_diameter / num_teeth). "
            "- pressure_angle_deg: pressure angle in degrees (standard: 20.0). "
            "- thickness: face width / axial thickness in mm. "
            "- bore_diameter: central hole diameter in mm (0 = no bore). "
            "- pitch_diameter: if provided, overrides module (module = pitch_diameter / num_teeth). "
            "This function returns a cq.Workplane with a proper involute tooth profile. "
            "Assign its return value directly to 'result'. "
            "\n"
            "CRITICAL CADQUERY API RULES (anti-hallucination): "
            "(A) NEVER invent or guess CadQuery methods that do not exist. "
            "The following methods DO NOT EXIST in CadQuery and must NEVER be used: "
            "rotate_about_origin(), revolve_about(), sweep_along(), make_gear(), "
            "make_thread(). If you are unsure whether a method exists, "
            "use basic operations instead. "
            "(B) Only use standard CadQuery directional selectors: "
            "'>Z' (top), '<Z' (bottom), '>X' (right), '<X' (left), '>Y' (front), '<Y' (back). "
            "NEVER invent face names like 'face_top_horizontal'. "
            "(C) For placing holes or features at specific positions, create a new "
            "cq.Workplane at an explicit coordinate offset, e.g.: "
            "result = result.faces('>Z').workplane().center(x, y).circle(r).cutThruAll() "
            "(D) Build complex parts from simple boolean operations (union/cut) on "
            "basic primitives (box, cylinder). Avoid complex single-sketch extrusions. "
            "\n"
            "GEOMETRY SIMPLIFICATION RULES: "
            "(E) For THREADS: Approximate using a plain cylinder with the correct "
            "dimensions. Do NOT attempt helical sweeps. "
            "(F) For ANGULAR/FACETED curved geometry (e.g. star profiles, irregular polygons): "
            "use polygon approximations with math.cos and math.sin, and "
            "cq.Workplane().polyline(points).close().extrude(). "
            "EXCEPTION — ORGANIC/SMOOTH shapes (described as 'smooth', 'ergonomic', "
            "'mouse-like', 'curved body'): you are STRICTLY FORBIDDEN from using "
            ".polyline() for these. Instead construct cross-sections using .spline(), "
            ".ellipse(), or .circle() to produce smooth BRep geometry. "
            "(G) POLAR ARRAY RULE (MANDATORY for circular patterns): "
            "When the description mentions 'bolt circle', 'holes evenly spaced in a circle', "
            "'circular pattern', 'radial pattern', or any N features equally spaced around "
            "a center, you MUST use CadQuery's .polarArray() method. "
            "Do NOT manually compute positions with sin/cos loops. "
            "Pattern: .faces('>Z').workplane() "
            ".polarArray(radius=R, startAngle=0, angle=360, count=N).hole(D) "
            "where R is the bolt circle radius (60-75% of the part outer radius), "
            "N is the feature count, and D is the hole diameter. "
            "\n"
            "LOFT RULE (MANDATORY for organic/tapered shapes): "
            "When the description is a vase, bottle, funnel, cone, or any shape with "
            "cross-sections that vary along an axis, you MUST use the .loft() approach. "
            "Draw the first cross-section, call .workplane(offset=Z) to step up the "
            "Z-axis, draw the second cross-section, then call .loft(). "
            "Example: result = cq.Workplane('XY').circle(25).workplane(offset=80).circle(15).loft() "
            "For polygonal bases: result = cq.Workplane('XY').polygon(nSides=6, diameter=50).workplane(offset=150).circle(35).loft() "
            "CRITICAL: You MUST call .workplane(offset=Z) with a non-zero Z between "
            "cross-sections. If both sections are on the same Z-plane, .loft() will "
            "raise 'ValueError: Nothing to loft'. "
            "TWISTING LOFTS EXCEPTION: To twist a cross-section between loft steps, "
            "do NOT call .rotate() mid-chain (causes BRep_API errors). Instead, rotate "
            "the local coordinate system of the new workplane BEFORE drawing the shape: "
            "cq.Workplane('XY').rect(10,10).workplane(offset=50).transformed(rotate=cq.Vector(0,0,45)).rect(10,10).loft() "
            "\n"
            "STRICT LOFT CHAINING (CRASH PREVENTION): "
            "You are STRICTLY FORBIDDEN from passing shape variables into .loft(). "
            "NEVER do: s1 = wp.rect(...); s2 = wp2.circle(...); s1.loft(s2) — this "
            "causes a ValueError. ALL lofts MUST be a single continuous chain on one "
            "cq.Workplane object: cq.Workplane('XY').rect(10,10).workplane(offset=10).circle(5).loft(). "
            "\n"
            "HOLLOWING / SHELLING RULE (CRASH PREVENTION): "
            "Do NOT hollow out organic/complex parts by boolean-subtracting a slightly "
            "smaller copy of the shape from itself (Russian Doll anti-pattern). This "
            "produces fragile non-manifold geometry that crashes the BRep kernel. "
            "MANDATORY: Select the face you want to open and use .shell(thickness). "
            "Example for a vase: result = [loft chain].faces('>Z').shell(-2.0) "
            "\n"
            "NO HALLUCINATED 2D METHODS (CRASH PREVENTION): "
            "The methods .fillet2D() and .chamfer2D() DO NOT EXIST in CadQuery. "
            "NEVER use them. To round or chamfer edges: create the 3D solid first "
            "(via .extrude() or .loft()), then apply .edges('|Z').fillet(r) or "
            ".edges('|Z').chamfer(r) on the 3D solid. "
            "\n"
            "SAFE FILLETING (CRASH PREVENTION): "
            "NEVER hardcode arbitrary large fillet radii. Large radii on thin walls "
            "crash OpenCASCADE with StdFail_NotDone. When the prompt asks for "
            "smooth/ergonomic edges without specifying an exact radius, default to "
            "1.0 or 2.0. RULE: fillet radius MUST be less than half of the thinnest "
            "wall or feature dimension of the part. "
            "\n"
            "EXPLICIT FACE RE-SELECTION AFTER BOOLEANS (CRASH PREVENTION): "
            "After any .cut() or .union() operation, the workplane context stack may "
            "reference a stale or destroyed face. You MUST explicitly re-select a face "
            "before initializing a new sketch. "
            "CORRECT: part = base.cut(pocket); part = part.faces('>Z').workplane().circle(5).extrude(10) "
            "WRONG: part = base.cut(pocket).circle(5).extrude(10)"
            "\n"
            "CRITICAL CADQUERY CONSTRAINTS: "
            "(1) 2D-TO-3D FOR PRISMATICS: NEVER assemble brackets, flanges, or L-shapes "
            "by unioning separate 3D boxes. You MUST sketch the full continuous 2D "
            "footprint using lines/arcs on a single Workplane and execute a single "
            ".extrude(). "
            "(2) ORGANIC SHAPES: For 'smooth' or 'ergonomic' shapes (like a mouse), you "
            "are STRICTLY FORBIDDEN from using .polyline(). You must construct "
            "cross-sections using .spline(), .ellipse(), or .circle(). "
            "(3) TWISTING LOFTS: Do not use .rotate() in the middle of a loft chain "
            "(causes BRep_API errors). To twist a cross-section, rotate the local "
            "coordinate system of the new workplane BEFORE drawing the shape using: "
            "cq.Workplane('XY').rect(10,10).workplane(offset=50)"
            ".transformed(rotate=cq.Vector(0,0,45)).rect(10,10).loft(). "
            "(4) HOLLOWING: Use .shell(-thickness) on a face to hollow parts. "
            "(5) NO 2D FILLETS: .fillet() on 2D sketches causes a ValueError. Apply "
            ".fillet() ONLY to 3D solids after .loft() or .extrude(). "
            "(6) CLOSED LOOPS (Möbius): A continuous ring must not be lofted straight up "
            "the Z-axis. You must use .sweep() along a circular path."
        ),
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


async def run_critic_loop(part_def: PartDefinition, max_retries: int = 3) -> str:
    """Subagent 3: The QA Inspector (Critic Loop).

    Runs the Machinist's code via subprocess with concurrency control.
    On failure, feeds the traceback back for self-correction.
    """
    async with _MACHINIST_SEMAPHORE:
        error_context = ""

        for attempt in range(1, max_retries + 1):
            code = await run_machinist(part_def, error_context=error_context)
            result = await execute_cad_script(code)

            if result["status"] == "success":
                return code

            error_context = result["traceback"]
            if attempt == max_retries:
                raise ScriptError(
                    f"Part '{part_def.part_id}' failed after {max_retries} attempts. "
                    f"Last error:\n{error_context}",
                    script=code,
                )

        raise ScriptError(f"Part '{part_def.part_id}' failed unexpectedly.")


async def run_machinist_batch(parts: list[PartDefinition]) -> dict[str, str]:
    """Run all Machinist critic loops concurrently with rate limiting."""
    tasks = [run_critic_loop(part) for part in parts]
    results = await asyncio.gather(*tasks)
    return {part.part_id: code for part, code in zip(parts, results)}


async def run_single_part_export(part_script: str, output_filename: str) -> str:
    """Generate a simple export script for single-part models (no assembly)."""
    return (
        f"{part_script}\n\n"
        f"import cadquery as cq\n"
        f"assembly = cq.Assembly()\n"
        f"assembly.add(result, name='part')\n"
        f"assembly.save('{output_filename}')\n"
    )


async def run_assembler(
    manifest: AssemblyManifest, part_scripts: dict[str, str],
    error_context: str = "",
) -> str:
    """Subagent 4: The Assembler.

    Takes all validated part scripts and the mating rules, then writes a
    single final CadQuery script that assembles everything and exports to .glb.
    """
    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=(
            "You are the Assembler for a mechanical CAD system. Your job is to "
            "write a single, final CadQuery Python script that combines multiple "
            "individual parts into one assembly using cq.Assembly(). "
            "\n"
            "RULES: "
            "(1) Import cadquery as cq at the top. "
            "(2) Paste each part's code inline (do NOT use file imports). Each "
            "part script defines a variable named 'result'. Rename each to a "
            "unique variable (e.g., base_plate, leg_1) immediately after pasting. "
            "(3) Create the assembly: assembly = cq.Assembly() "
            "(4) Add each part using the EXACT translation coordinates from the "
            "mating rules. The 'translation' field contains 'X, Y, Z' values. "
            "Parse them and use: "
            "assembly.add(part_var, name='part_id', loc=cq.Location(cq.Vector(X, Y, Z))) "
            "The base/first part is always at (0, 0, 0). "
            "\n"
            "CRITICAL — DO NOT use assembly.constrain() or assembly.solve(). "
            "DO NOT try to compute positions yourself. Use ONLY the translation "
            "coordinates provided in the mating rules. They are pre-computed. "
            "\n"
            "(5) End the script with: assembly.save('output.glb') "
            "(6) Return ONLY raw executable Python. No markdown, no code fences. "
            "(7) If there is only ONE part, simply create the assembly, add it at "
            "the origin, and save."
        ),
    )

    parts_section = ""
    for part_id, code in part_scripts.items():
        parts_section += f"\n--- Part: {part_id} ---\n{code}\n"

    rules_section = ""
    for rule in manifest.mating_rules:
        rules_section += (
            f"  - Place '{rule.target_part_id}' at translation ({rule.translation}) "
            f"[mating {rule.source_part_id}@{rule.source_anchor} "
            f"to {rule.target_part_id}@{rule.target_anchor}, "
            f"clearance: {rule.clearance}mm]\n"
        )

    prompt = (
        f"Assembly Name: {manifest.assembly_name}\n\n"
        f"== VALIDATED PART SCRIPTS ==\n{parts_section}\n"
        f"== MATING RULES ==\n{rules_section}\n"
        f"Write the final assembly script."
    )

    if error_context:
        prompt += (
            f"\nYour previous assembly script FAILED with this error:\n"
            f"---\n{error_context}\n---\n"
            f"Fix the script to resolve this error. "
            f"Return the complete corrected script."
        )

    response = await model.generate_content_async(prompt)
    return _strip_markdown_fences(response.text)


async def run_assembly_critic_loop(
    manifest: AssemblyManifest, part_scripts: dict[str, str], max_retries: int = 3,
) -> str:
    """Critic loop for the Assembler — retries on execution failure."""
    error_context = ""

    for attempt in range(1, max_retries + 1):
        code = await run_assembler(manifest, part_scripts, error_context=error_context)
        result = await execute_cad_script(code)

        if result["status"] == "success":
            return code

        error_context = result["traceback"]
        if attempt == max_retries:
            raise ScriptError(
                f"Assembly failed after {max_retries} attempts. "
                f"Last error:\n{error_context}",
                script=code,
            )

    raise ScriptError("Assembly failed unexpectedly.")
