# Text-to-CAD Engine: Architecture & Generation Rules

## 1. Core Topological Requirements
* **Strict Manifold Solids:** The generation engine must ALWAYS output a valid, watertight `cq.Solid`.
* **No Floating Faces:** If a CadQuery operation results in a `cq.Compound` of disconnected, zero-thickness `cq.Face` or `cq.Wire` objects, it is a critical failure.
* **Validation:** All generation functions should verify the resulting object is a solid geometry before passing the result to the exporter.

## 2. Geometry Routing Strategy
The engine's logic must classify user text prompts into one of three domains and strictly apply the corresponding CadQuery generation strategy:

### Domain A: Primitives & Structural (e.g., boxes, simple plates, nuts)
* **Method:** Use basic 3D primitives (`.box()`, `.cylinder()`) combined with simple boolean operations (`.cut()`, `.union()`) and standard `.fillet()`/`.chamfer()`.

### Domain B: Mechanical & Parametric (e.g., spur gears, brackets, circular flanges)
* **Method:** STRICT 2D-to-3D Extrusion Workflow.
* **Rule:** Do NOT assemble complex mechanical parts by adding/subtracting 3D chunks.
* **Execution:** Sketch the fully closed 2D continuous boundary (and inner holes/bores) on a single `cq.Workplane`, then perform a single `.extrude()` operation.
* **Patterning:** Strictly use CadQuery's native mathematical patterning methods (e.g., `.polarArray()`) for repeated features like bolt circles. Do not manually calculate Cartesian coordinates for radial patterns.

### Domain C: Organic & Ergonomic (e.g., vases, handles)
* **Method:** Profile Lofting.
* **Rule:** Do NOT use extreme filleting on basic primitives to approximate organic curvature.
* **Execution:** Create a 2D cross-section, step along an axis using an offset (e.g., `.workplane(offset=Z)`), sketch the next cross-section, and execute `.loft()`.

## 3. OpenCASCADE Crash Prevention Directives
The following rules prevent the most common runtime crashes (topology errors, BRep failures, ValueErrors) in the OpenCASCADE kernel used by CadQuery.

### Directive 1 — Strict Chaining for Lofts (No Variables)
* **Problem:** Passing separate shape variables into `.loft()` (e.g., `shape1.loft(shape2)`) causes a `ValueError`.
* **Mandatory Pattern:** All lofts MUST be chained on a single continuous `cq.Workplane` object stack. Never assign intermediate cross-sections to separate variables.
* **Example:**
  ```python
  result = (
      cq.Workplane("XY")
      .rect(10, 10)
      .workplane(offset=10)
      .circle(5)
      .loft()
  )
  ```

### Directive 2 — Hollowing / Shelling (No Russian Dolls)
* **Problem:** Boolean-subtracting a slightly smaller copy of a complex/organic shape from itself produces fragile, non-manifold geometry that crashes the BRep kernel.
* **Mandatory Pattern:** Select the face you want to open and use the native `.shell(thickness)` method.
* **Example (vase):**
  ```python
  result = (
      cq.Workplane("XY")
      .circle(30)
      .workplane(offset=100)
      .circle(20)
      .loft()
      .faces(">Z").shell(-2.0)
  )
  ```

### Directive 3 — No Hallucinated 2D Methods
* **Problem:** Methods like `.fillet2D()` and `.chamfer2D()` do not exist in CadQuery. Using them raises `AttributeError`.
* **Mandatory Pattern:** Perform standard 2D sketches (`.rect()`, `.polygon()`), extrude/loft to a 3D solid, and then apply 3D edge operations explicitly (e.g., `.edges("|Z").fillet(radius)`).

### Directive 4 — Safe Filleting (Prevent BRep Crashes)
* **Problem:** Hardcoded large fillet radii crash the OpenCASCADE topology engine on thin walls or small features (`StdFail_NotDone`).
* **Mandatory Pattern:** When a prompt requests smooth or ergonomic edges without specifying an exact radius, default to a safe radius (`1.0` or `2.0`). Mathematically ensure `radius < (thinnest_dimension / 2)`.

### Directive 5 — Explicit Face Re-Selection After Booleans
* **Problem:** After `.cut()` or `.union()`, the CadQuery context stack may reference a stale face. Starting a new sketch on that context produces topology errors.
* **Mandatory Pattern:** Always explicitly re-select a face before initializing a new sketch after any boolean operation.
* **Example:**
  ```python
  part = base.cut(pocket)
  part = part.faces(">Z").workplane().circle(5).extrude(10)
  ```

## 4. Data Pipeline & UI Binding
* **API Payload:** The backend generation endpoint must always return a payload containing BOTH the compiled 3D model data AND the raw CadQuery Python script string.
* **UI Component:** The frontend must consistently bind the returned raw code string to the "CadQuery Code" viewer window. This must execute even if the 3D build fails, ensuring the code is exposed for debugging.

## 6. Gemini API Prompt Engineering

The following 6 constraints are injected verbatim into the Machinist agent's `system_instruction` on every Gemini API call. They represent hard-won rules that prevent specific geometric and topological failures discovered in production.

### Rule 1 — 2D-to-3D for Prismatics (No Box Unions)
* **Problem:** Assembling brackets, flanges, or L-shapes by unioning separate 3D boxes produces non-manifold seams and inconsistent edge topology.
* **Mandatory Pattern:** Sketch the full continuous 2D footprint of the part (using lines/arcs) on a **single `Workplane`** and execute a single `.extrude()`.

### Rule 2 — Organic Shapes (No `.polyline()`)
* **Problem:** `.polyline()` generates faceted, angular geometry. Using it for smooth or ergonomic shapes (like a mouse body) produces obvious flat faces instead of smooth curvature.
* **Mandatory Pattern:** For any shape described as "smooth", "ergonomic", or "organic", construct cross-sections using `.spline()`, `.ellipse()`, or `.circle()` — **`.polyline()` is strictly forbidden** for these shapes.

### Rule 3 — Twisting Lofts (No Mid-Chain `.rotate()`)
* **Problem:** Calling `.rotate()` mid-loft-chain causes `BRep_API` errors because the context workplane has already been consumed.
* **Mandatory Pattern:** To twist a cross-section between loft steps, rotate the **local coordinate system of the new workplane** using `.transformed()` **before** drawing the shape:
  ```python
  result = (
      cq.Workplane("XY")
      .rect(10, 10)
      .workplane(offset=50)
      .transformed(rotate=cq.Vector(0, 0, 45))
      .rect(10, 10)
      .loft()
  )
  ```

### Rule 4 — Hollowing (`.shell()` Only)
* **Problem:** Boolean-subtracting a smaller copy of an organic shape from itself ("Russian Doll") creates non-manifold geometry that crashes the BRep kernel.
* **Mandatory Pattern:** Select the face to open and apply `.shell(-thickness)`.

### Rule 5 — No 2D Fillets
* **Problem:** Calling `.fillet()` on a 2D sketch wire (e.g., directly after `.rect()`) raises a `ValueError` because no solid edges exist yet.
* **Mandatory Pattern:** Apply `.fillet()` **only** to 3D solids — after `.loft()` or `.extrude()`.

### Rule 6 — Closed Loops / Möbius Rings (`.sweep()` Required)
* **Problem:** Lofting a ring cross-section straight up the Z-axis does not close back on itself — it produces an open tube, not a torus or continuous ring.
* **Mandatory Pattern:** A continuous closed loop **must** use `.sweep()` along a circular or closed spline path. Do **not** attempt to close it with a loft.

---

## 5. Assistant Directives (For Claude Code)
* **Diff-Only Updates:** When modifying existing files, output ONLY the modified functions or classes. Do not rewrite unchanged code.
* **No Filler:** Output raw code solutions and technical explanations directly. Omit conversational filler to conserve tokens.