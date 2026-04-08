"""
Parametric CAD utility functions for the Text-to-CAD pipeline.

These functions are importable by Machinist-generated scripts via PYTHONPATH.
They provide mathematically correct geometry for parts that LLMs consistently
fail to generate (gears, threads, etc.).
"""

import math

import cadquery as cq


def make_involute_spur_gear(
    num_teeth: int = 20,
    module: float = 2.0,
    pressure_angle_deg: float = 20.0,
    thickness: float = 10.0,
    bore_diameter: float = 0.0,
    pitch_diameter: float | None = None,
) -> cq.Workplane:
    """Generate a mathematically accurate involute spur gear as a 3D solid.

    Uses a 2D sketch approach: computes the full gear tooth profile as a
    single continuous closed wire, adds the bore as an inner wire, and
    performs a single extrude(). This avoids repeated boolean subtractions
    which can fragment geometry into a non-manifold Compound.

    The tooth profile is computed from the standard involute function:
        inv(alpha) = tan(alpha) - alpha

    Each tooth consists of:
        - Left involute flank  (base/root circle -> addendum circle)
        - Tip arc              (along addendum circle)
        - Right involute flank (addendum circle -> base/root circle)
        - Root arc             (along root circle to the next tooth)

    Args:
        num_teeth: Number of teeth (Z). Must be >= 6.
        module: Gear module in mm (tooth size). Ignored if pitch_diameter is set.
        pressure_angle_deg: Pressure angle in degrees (standard: 20).
        thickness: Axial thickness (face width) in mm.
        bore_diameter: Central bore hole diameter in mm. 0 = no bore.
                       Must be less than the root circle diameter.
        pitch_diameter: If provided, overrides module. module = pitch_diameter / num_teeth.

    Returns:
        cq.Workplane containing the extruded 3D gear solid, centered at origin.

    Raises:
        ValueError: If bore_diameter >= root circle diameter, or if the
                    result is not a manifold cq.Solid.
    """
    if pitch_diameter is not None:
        module = pitch_diameter / num_teeth

    pa = math.radians(pressure_angle_deg)

    # Standard gear circle radii
    pitch_r = num_teeth * module / 2.0
    base_r = pitch_r * math.cos(pa)
    outer_r = pitch_r + module                        # addendum circle
    root_r = max(pitch_r - 1.25 * module, 0.5)       # dedendum circle

    if bore_diameter > 0 and bore_diameter >= 2.0 * root_r:
        raise ValueError(
            f"bore_diameter ({bore_diameter}) must be less than "
            f"root circle diameter ({2.0 * root_r:.3f})"
        )

    def involute_r_theta(t: float) -> tuple[float, float]:
        """Radius and polar angle offset on the involute at parameter t."""
        r = base_r * math.sqrt(1.0 + t * t)
        theta = t - math.atan(t)
        return r, theta

    def t_at_radius(r: float) -> float:
        """Involute parameter t at a given radius."""
        if r <= base_r:
            return 0.0
        return math.sqrt((r / base_r) ** 2 - 1.0)

    # Half angular tooth thickness at pitch circle
    inv_pa = math.tan(pa) - pa
    half_tooth_angle = math.pi / (2.0 * num_teeth) + inv_pa

    angular_pitch = 2.0 * math.pi / num_teeth
    n_flank = 15       # points per involute flank
    n_tip = 3          # interpolation points on tip arc
    n_root = 3         # interpolation points on root arc

    t_tip = t_at_radius(outer_r)
    t_start = t_at_radius(max(root_r, base_r))
    _, dtheta_start = involute_r_theta(t_start)
    _, dtheta_tip = involute_r_theta(t_tip)

    # ── Build full 2D gear profile ──────────────────────────────────
    # Traced counterclockwise around the gear.  For each tooth:
    #   left flank outward → tip arc → right flank inward → root arc
    profile_pts: list[tuple[float, float]] = []

    for i in range(num_teeth):
        theta = i * angular_pitch

        # -- Left involute flank (outward: root/base → tip) --
        if root_r < base_r:
            a = theta - half_tooth_angle
            profile_pts.append((root_r * math.cos(a), root_r * math.sin(a)))

        for j in range(n_flank + 1):
            t = t_start + (t_tip - t_start) * j / n_flank
            r, dt = involute_r_theta(t)
            r = min(r, outer_r)
            a = theta - half_tooth_angle + dt
            profile_pts.append((r * math.cos(a), r * math.sin(a)))

        # -- Tip arc (left tip → right tip, counterclockwise) --
        a_left_tip = theta - half_tooth_angle + dtheta_tip
        a_right_tip = theta + half_tooth_angle - dtheta_tip
        for j in range(1, n_tip + 1):
            frac = j / (n_tip + 1)
            a = a_left_tip + (a_right_tip - a_left_tip) * frac
            profile_pts.append((outer_r * math.cos(a), outer_r * math.sin(a)))

        # -- Right involute flank (inward: tip → root/base) --
        for j in range(n_flank + 1):
            t = t_tip - (t_tip - t_start) * j / n_flank
            r, dt = involute_r_theta(t)
            r = min(r, outer_r)
            a = theta + half_tooth_angle - dt
            profile_pts.append((r * math.cos(a), r * math.sin(a)))

        if root_r < base_r:
            a = theta + half_tooth_angle
            profile_pts.append((root_r * math.cos(a), root_r * math.sin(a)))

        # -- Root arc to next tooth --
        a_end = theta + half_tooth_angle - dtheta_start
        if root_r < base_r:
            a_end = theta + half_tooth_angle

        next_theta = (i + 1) * angular_pitch
        a_next = next_theta - half_tooth_angle + dtheta_start
        if root_r < base_r:
            a_next = next_theta - half_tooth_angle

        for j in range(1, n_root + 1):
            frac = j / (n_root + 1)
            a = a_end + (a_next - a_end) * frac
            profile_pts.append((root_r * math.cos(a), root_r * math.sin(a)))

    # ── Extrude the single closed wire (+ bore) ────────────────────
    wp = cq.Workplane("XY").polyline(profile_pts).close()
    if bore_diameter > 0:
        wp = wp.circle(bore_diameter / 2.0)
    result = wp.extrude(thickness)

    # Center vertically on the origin
    result = result.translate((0, 0, -thickness / 2.0))

    # ── Topological validation ──────────────────────────────────────
    solid = result.val()
    if not isinstance(solid, cq.Solid):
        raise ValueError(
            f"Topological validation failed: expected cq.Solid, "
            f"got {type(solid).__name__}."
        )

    return result


def make_naca_wire(
    naca_code: str = "2412",
    chord_length: float = 200.0,
) -> cq.Wire:
    """Generate a NACA 4-digit airfoil as a closed 2D cq.Wire.

    Computes exact upper and lower surface coordinates from the standard
    NACA 4-digit analytic thickness and camber distributions, builds two
    B-spline edges (upper and lower surfaces) via OpenCASCADE's
    Geom_BSplineCurve interpolation, and assembles them into a single
    closed, manifold cq.Wire. The wire begins and ends precisely at the
    trailing edge and lies in the XY plane at Z=0.

    This wire is the correct primitive for spanwise lofting. Position it
    at a global Z station with wire.translate(cq.Vector(0, 0, z)) and
    pass multiple positioned wires to cq.Solid.makeLoft().

    NACA 4-digit encoding:
        - Digit 1: max camber as percent of chord  (m = d1 / 100)
        - Digit 2: max camber location in tenths   (p = d2 / 10)
        - Digits 3-4: max thickness as percent      (t = d34 / 100)

    Coordinate system:
        - X axis: chordwise, centered (LE at -chord/2, TE at +chord/2)
        - Y axis: thickness direction
        - Wire lies in the XY plane at Z = 0

    Uses the modified trailing-edge coefficient (-0.1036 for the x^4 term)
    to produce a mathematically closed trailing edge (zero TE gap), ensuring
    the wire is watertight without a separate closing edge.

    Args:
        naca_code: 4-digit NACA designation string (e.g., "2412", "0012").
        chord_length: Chord length in mm.

    Returns:
        A closed cq.Wire in the XY plane at Z=0, centered on the chord
        midpoint. Starts and ends at the trailing edge (+chord/2, 0, 0).

    Raises:
        ValueError: If naca_code is not exactly 4 digits.
    """
    if len(naca_code) != 4 or not naca_code.isdigit():
        raise ValueError(
            f"Invalid NACA code '{naca_code}': must be exactly 4 digits "
            f"(e.g., '2412', '0012')."
        )

    m = int(naca_code[0]) / 100.0   # max camber
    p = int(naca_code[1]) / 10.0    # max camber position
    t = int(naca_code[2:4]) / 100.0  # max thickness ratio

    # ── Compute surface coordinates ────────────────────────────────
    # Cosine spacing concentrates points near the leading edge where
    # curvature is highest, improving spline fidelity.
    n_pts = 80
    upper_vecs: list[cq.Vector] = []
    lower_vecs: list[cq.Vector] = []

    for i in range(n_pts + 1):
        beta = i * math.pi / n_pts
        x = (1.0 - math.cos(beta)) / 2.0  # normalized [0, 1]

        # Half-thickness distribution (modified form: closed trailing edge)
        yt = (t / 0.20) * (
            0.2969 * math.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x * x
            + 0.2843 * x * x * x
            - 0.1036 * x * x * x * x
        )

        # Camber line and local slope
        if m == 0.0 or p == 0.0:
            yc = 0.0
            theta = 0.0
        else:
            if x < p:
                yc = (m / (p * p)) * (2.0 * p * x - x * x)
                dyc_dx = (2.0 * m / (p * p)) * (p - x)
            else:
                yc = (m / ((1.0 - p) ** 2)) * (
                    (1.0 - 2.0 * p) + 2.0 * p * x - x * x
                )
                dyc_dx = (2.0 * m / ((1.0 - p) ** 2)) * (p - x)
            theta = math.atan(dyc_dx)

        # Upper and lower surface points (perpendicular to camber line)
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        upper_vecs.append(cq.Vector(
            (x - yt * sin_t) * chord_length,
            (yc + yt * cos_t) * chord_length,
            0.0,
        ))
        lower_vecs.append(cq.Vector(
            (x + yt * sin_t) * chord_length,
            (yc - yt * cos_t) * chord_length,
            0.0,
        ))

    # ── Build closed profile wire via OCCT B-spline interpolation ──
    # Upper edge: trailing edge → leading edge (reversed point order)
    # Lower edge: leading edge → trailing edge (natural order)
    # Both edges share exact LE and TE endpoints (yt=0 at x=0 and x=1),
    # forming a mathematically closed loop: TE → LE (upper) → TE (lower).
    upper_edge = cq.Edge.makeSpline(list(reversed(upper_vecs)))
    lower_edge = cq.Edge.makeSpline(lower_vecs)

    wire = cq.Wire.assembleEdges([upper_edge, lower_edge])

    # Center on chord midpoint (LE at -chord/2, TE at +chord/2)
    wire = wire.translate(cq.Vector(-chord_length / 2.0, 0.0, 0.0))

    return wire


def make_naca_airfoil(
    naca_code: str = "2412",
    chord_length: float = 200.0,
    span: float = 500.0,
) -> cq.Workplane:
    """Generate a NACA 4-digit airfoil as an extruded 3D solid (B-Rep volume).

    Delegates to make_naca_wire() for the mathematically exact 2D profile,
    then extrudes linearly along the span axis and centers the result at
    the origin.

    NACA 4-digit encoding:
        - Digit 1: max camber as percent of chord  (m = d1 / 100)
        - Digit 2: max camber location in tenths   (p = d2 / 10)
        - Digits 3-4: max thickness as percent      (t = d34 / 100)

    Coordinate system:
        - X axis: chordwise (leading edge at -chord/2, trailing edge at +chord/2)
        - Y axis: thickness direction
        - Z axis: spanwise (centered at origin)

    Args:
        naca_code: 4-digit NACA designation string (e.g., "2412", "0012").
        chord_length: Chord length in mm.
        span: Spanwise extrusion length in mm.

    Returns:
        cq.Workplane containing the extruded 3D airfoil solid, centered
        at the origin.

    Raises:
        ValueError: If naca_code is not exactly 4 digits, or if the result
                    is not a manifold cq.Solid.
    """
    wire = make_naca_wire(naca_code, chord_length)

    # ── Extrude along span (Z-axis) ───────────────────────────────
    solid = cq.Solid.extrudeLinear(wire, [], cq.Vector(0, 0, span))

    # Center span on origin (wire is already chord-centered)
    result = cq.Workplane("XY").newObject([solid])
    result = result.translate((0.0, 0.0, -span / 2.0))

    # ── Topological validation ─────────────────────────────────────
    val = result.val()
    if not isinstance(val, cq.Solid):
        raise ValueError(
            f"NACA airfoil validation failed: expected cq.Solid, "
            f"got {type(val).__name__}."
        )

    return result


def make_metric_thread(
    diameter: float,
    pitch: float,
    length: float,
) -> cq.Workplane:
    """Generate a stable ribbed-cylinder approximation of a metric thread.

    Per Domain B rules, a true 3D helical sweep crashes the BRep kernel.
    Instead, this builds a cylindrical shaft with evenly spaced annular ribs
    that approximate thread crests. The result is a manifold solid that
    OpenCASCADE can reliably boolean-union or boolean-cut.

    Args:
        diameter: Nominal (major) thread diameter in mm.
        pitch: Thread pitch in mm (distance between crests).
        length: Total thread length in mm.

    Returns:
        cq.Workplane solid centered at the origin.
    """
    major_r = diameter / 2.0
    minor_r = major_r - 0.6134 * pitch  # standard 60° thread depth
    rib_height = major_r - minor_r
    rib_width = pitch * 0.4

    # Core cylinder at the minor diameter
    result = cq.Workplane("XY").circle(minor_r).extrude(length)

    # Stack annular ribs along the length to simulate thread crests
    num_ribs = int(length / pitch)
    for i in range(num_ribs):
        z_offset = pitch * 0.5 + i * pitch
        if z_offset + rib_width / 2.0 > length:
            break
        rib = (
            cq.Workplane("XY")
            .workplane(offset=z_offset - rib_width / 2.0)
            .circle(major_r)
            .circle(minor_r)
            .extrude(rib_width)
        )
        result = result.union(rib)

    # Center on origin
    result = result.translate((0, 0, -length / 2.0))

    solid = result.val()
    if not isinstance(solid, cq.Solid):
        raise ValueError(
            f"Thread validation failed: expected cq.Solid, "
            f"got {type(solid).__name__}."
        )

    return result
