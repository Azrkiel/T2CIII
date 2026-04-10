"""
Constraint-Based Assembly Schemas for Text-to-CAD Pipeline.

These Pydantic models define the strict inter-agent communication contract.
They are used as `response_schema` in Gemini API calls to enforce structured,
hallucination-resistant output from every subagent.

CORE RULE: All spatial relationships are TOPOLOGICAL and RELATIVE.
Absolute floating-point coordinates (x, y, z) are FORBIDDEN for mating.
Parts are connected via named anchor tags that reference CadQuery selectors.
"""

from pydantic import BaseModel, Field, model_validator


class PartDefinition(BaseModel):
    """A single mechanical part to be manufactured by the Machinist subagent.

    Each part is generated in COMPLETE ISOLATION — the Machinist receives only
    this definition and must produce a self-contained CadQuery script that
    creates the part centered at the origin. The part's relationship to the
    broader assembly is handled entirely by anchor_tags and MatingRules —
    the Machinist must NEVER embed assembly-level positioning logic.

    EXCEPTION — Coupled Geometric Subsystems (Domain D Aerospace):
    When a part represents a tightly coupled aerodynamic subsystem (e.g., a
    wing segment with internal ribs and spars), the Machinist is permitted —
    and REQUIRED — to generate the entire subsystem within a single self-
    contained script. This is necessary because internal structures depend on
    a shared tooling body (the ``inner_void`` — the directly-lofted airfoil
    core solid). Splitting such a subsystem across multiple PartDefinition
    entries makes it impossible to share that master geometry, causing ribs
    to clip through shell walls. In these cases the Machinist must produce a
    ``cq.Assembly`` containing all sub-bodies (skin, ribs, spars) and assign
    it to ``result``.
    """

    part_id: str = Field(
        ...,
        max_length=200,
        description=(
            "A unique, snake_case identifier for this part (e.g., 'base_plate', "
            "'motor_housing', 'm5_bolt'). This ID is referenced by MatingRules "
            "to connect parts in the assembly. Must be unique across the entire "
            "AssemblyManifest."
        ),
    )

    description: str = Field(
        ...,
        max_length=2000,
        description=(
            "A detailed physical description of the part that the Machinist "
            "subagent will use to write CadQuery code. Include ALL geometric "
            "details: dimensions (in millimeters), shape primitives (box, "
            "cylinder, etc.), features (holes, fillets, chamfers, slots), and "
            "material intent. Be explicit and unambiguous. "
            "Example: 'A 100mm x 60mm x 5mm rectangular base plate with four "
            "M5 through-holes at each corner, 8mm from each edge, and a 2mm "
            "fillet on all top edges.' "
            "DO NOT include any absolute (x, y, z) positioning relative to "
            "other parts. Each part is modeled at the origin in isolation."
        ),
    )

    anchor_tags: list[str] = Field(
        ...,
        description=(
            "A list of named anchors on this part where other parts can attach. "
            "Each anchor tag MUST be one of the following: "
            "(1) A valid CadQuery face/edge selector string such as '>Z' (top "
            "face), '<Z' (bottom face), '>X' (right face), '<X' (left face), "
            "'>Y' (front face), '<Y' (back face). "
            "(2) A descriptive named tag for a specific feature, e.g., "
            "'hole_center', 'slot_midpoint', 'flange_top'. The Machinist will "
            "use cq.Assembly().tag() to register these in the CadQuery script. "
            "CRITICAL: Anchor tags are TOPOLOGICAL REFERENCES, not coordinates. "
            "NEVER use tuples like '(10, 20, 0)' or any numeric coordinate as "
            "an anchor tag. The Assembler subagent uses these tags to mate "
            "parts together via CadQuery's constraint system."
        ),
    )


class MatingRule(BaseModel):
    """A constraint that connects two parts and specifies placement coordinates.

    MatingRules define both the logical connection between parts (via anchors)
    and the precise 3D translation for the Assembler to position each part.
    """

    source_part_id: str = Field(
        ...,
        description=(
            "The part_id of the first part in this mating pair. Must exactly "
            "match a part_id defined in the AssemblyManifest.parts list."
        ),
    )

    source_anchor: str = Field(
        ...,
        description=(
            "The anchor tag on the source part where the connection originates. "
            "Must exactly match one of the anchor_tags defined in the source "
            "part's PartDefinition. Example: '>Z'."
        ),
    )

    target_part_id: str = Field(
        ...,
        description=(
            "The part_id of the second part in this mating pair. Must exactly "
            "match a part_id defined in the AssemblyManifest.parts list. "
            "Must be different from source_part_id."
        ),
    )

    target_anchor: str = Field(
        ...,
        description=(
            "The anchor tag on the target part where the connection terminates. "
            "Must exactly match one of the anchor_tags defined in the target "
            "part's PartDefinition. Example: '<Z'."
        ),
    )

    translation: str = Field(
        ...,
        description=(
            "The precise absolute 3D translation for the TARGET part as a string "
            "in the format 'X, Y, Z' (in millimeters). The Assembler will parse "
            "this and use cq.Location(cq.Vector(X, Y, Z)) to position the target "
            "part. The base/first part is always at '0, 0, 0'. Example: "
            "'-350, -200, -400' to place the target part at that world position."
        ),
    )

    clearance: float = Field(
        ...,
        description=(
            "Gap in millimeters between the two mated surfaces. "
            "Use 0.0 for flush contact. Use small positive values for "
            "mechanical clearance (e.g., 0.1 for a slip fit, 0.5 for a loose "
            "fit)."
        ),
    )


class AssemblyManifest(BaseModel):
    """The complete blueprint for a multi-part mechanical assembly.

    This is the output of the Draftsman (Planner) subagent and the input to
    the orchestration pipeline. It defines WHAT to build (parts) and HOW to
    connect them (mating_rules).

    The manifest is the single source of truth for the entire assembly. Every
    downstream subagent receives only its relevant slice of this manifest:
    - The Machinist gets one PartDefinition at a time.
    - The Assembler gets all validated part scripts + the mating_rules.

    ANTI-HALLUCINATION CONTRACT:
    - Parts must be self-contained and origin-centered.
    - All inter-part spatial relationships are encoded ONLY in mating_rules.
    - No absolute coordinates anywhere in this manifest.
    """

    assembly_name: str = Field(
        ...,
        max_length=200,
        description=(
            "A descriptive, snake_case name for the overall assembly "
            "(e.g., 'flanged_bearing_mount', 'gearbox_housing'). Used as the "
            "filename stem for the exported .glb file."
        ),
    )

    parts: list[PartDefinition] = Field(
        ...,
        description=(
            "An ordered list of all individual parts in the assembly. Each part "
            "will be manufactured independently by the Machinist subagent. "
            "Every part_id must be unique. List parts in logical build order "
            "(base/foundation first, fasteners last)."
        ),
    )

    mating_rules: list[MatingRule] = Field(
        ...,
        description=(
            "The list of constraints that define how parts connect to each "
            "other. Every mating rule references exactly two parts by their "
            "part_id and specifies which anchor_tags to mate. The Assembler "
            "subagent translates these into CadQuery constraint calls. "
            "RULES: "
            "(1) All referenced part_ids must exist in the parts list. "
            "(2) All referenced anchors must exist in the respective part's "
            "anchor_tags. "
            "(3) The assembly graph must be connected — no orphan parts. "
            "(4) NEVER encode absolute positions. Positioning is emergent "
            "from the chain of relative constraints."
        ),
    )

    @model_validator(mode="after")
    def _validate_manifest(self) -> "AssemblyManifest":
        # --- Size limits (not expressible in Gemini's schema) ---
        if len(self.parts) > 50:
            raise ValueError(
                f"Too many parts ({len(self.parts)}). Maximum is 50."
            )
        if len(self.mating_rules) > 100:
            raise ValueError(
                f"Too many mating rules ({len(self.mating_rules)}). Maximum is 100."
            )
        for part in self.parts:
            if len(part.anchor_tags) > 20:
                raise ValueError(
                    f"Part '{part.part_id}' has {len(part.anchor_tags)} "
                    f"anchor tags. Maximum is 20."
                )

        # --- Referential integrity ---
        valid_ids = {p.part_id for p in self.parts}
        for rule in self.mating_rules:
            if rule.source_part_id not in valid_ids:
                raise ValueError(
                    f"Mating rule references undefined part: {rule.source_part_id}"
                )
            if rule.target_part_id not in valid_ids:
                raise ValueError(
                    f"Mating rule references undefined part: {rule.target_part_id}"
                )
        return self
