"""
Anatomical Theatre, University of Virginia (Thomas Jefferson, 1826-1839).
Geometry reconstructed from the Historic American Buildings Survey sheets 1-7.

Authored in feet, exported in metres (glTF is specified in metres).
Origin sits at ground level, centre of the plan.  +Z is the front elevation.

Outputs:
    theatre.glb      complete exterior
    theatre_cut.glb  roof and two walls removed, interior exposed
"""
import math
import numpy as np
import trimesh
from trimesh.creation import extrude_polygon, box
from trimesh.transformations import rotation_matrix, translation_matrix, concatenate_matrices
from shapely.geometry import Polygon

FT = 0.3048

# ---- dimensions off the sheets -------------------------------------------
HALF, T, IH = 22.0, 1.5, 20.5      # 44'-0" square, 1'-6" brick, 20'-6" inside
Y_BOT, Y_MUS, Y_TH, Y_TOP = -1.0, 0.0, 10.83, 23.5
GROUND = -1.0                       # grade line; everything shifts up by this
CORN_H, PARA_H, EAVE = 2.17, 3.17, 1.4
WIN_R, SILL_LO, SILL_HI = 2.0, 7.5, 19.0
WIN_X = [-15.0, -6.0, 6.0, 15.0]    # 7'-0", 9'-0", 12'-0", 9'-0", 7'-0"
DOOR_W, DOOR_H = 4.0, 7.5
PIT, RUN, RISE, TIERS = 3.5, 3.0, 1.3333, 5
WELL = (-2.2, 2.2, 3.5, 15.5)       # centre stair well, x0 x1 z0 z1

# ---- palette: the Academical Village -------------------------------------
PAL = {
    "brick": (0.658, 0.337, 0.247),   # handmade Virginia brick
    "trim":  (0.937, 0.914, 0.859),   # off-white painted wood
    "stone": (0.612, 0.596, 0.541),
    "roof":  (0.490, 0.518, 0.471),   # painted tin
    "glass": (0.737, 0.847, 0.902),
    "dark":  (0.290, 0.251, 0.220),
    "wood":  (0.690, 0.541, 0.361),
}

# buckets, keyed by material; each side of the building is tagged so the
# cutaway can drop whole faces
BUCKETS = {}


def add(mat, mesh, side="core"):
    BUCKETS.setdefault((mat, side), []).append(mesh)


def T4(x=0.0, y=0.0, z=0.0):
    return translation_matrix([x, y, z])


def RY(a):
    return rotation_matrix(a, [0, 1, 0])


def RZ(a):
    return rotation_matrix(a, [0, 0, 1])


def bx(w, h, d, x, y, z, rot=None):
    m = box(extents=[w, h, d])
    xf = T4(x, y, z)
    if rot is not None:
        xf = concatenate_matrices(xf, rot)
    m.apply_transform(xf)
    return m


def arc(cx, cy, r, a0, a1, n=18):
    return [(cx + r * math.cos(a0 + (a1 - a0) * i / n),
             cy + r * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]


# ==========================================================================
# WALLS  — profile in local XY, extruded through the thickness in local Z
# ==========================================================================
def lunette(x, sill):
    return [(x + WIN_R, sill)] + arc(x, sill, WIN_R, 0, math.pi) + [(x - WIN_R, sill)]


def wall_polygon(door):
    outer = [(-HALF, Y_BOT), (HALF, Y_BOT), (HALF, Y_TOP), (-HALF, Y_TOP)]
    holes = []
    for x in WIN_X:
        holes.append(lunette(x, SILL_LO))
        holes.append(lunette(x, SILL_HI))
    if door:
        holes.append([(-DOOR_W / 2, Y_MUS), (DOOR_W / 2, Y_MUS),
                      (DOOR_W / 2, Y_MUS + DOOR_H), (-DOOR_W / 2, Y_MUS + DOOR_H)])
    return Polygon(outer, holes)


SIDES = [
    ("front", True,  concatenate_matrices(T4(0, 0, HALF - T))),
    ("rear",  True,  concatenate_matrices(T4(0, 0, -(HALF - T)), RY(math.pi))),
    ("left",  False, concatenate_matrices(T4(-HALF, 0, 0), RY(math.pi / 2))),
    ("right", False, concatenate_matrices(T4(HALF, 0, 0), RY(-math.pi / 2))),
]

for name, has_door, xf in SIDES:
    w = extrude_polygon(wall_polygon(has_door), T)
    w.apply_transform(xf)
    add("brick", w, name)

    # sashes: half-round frame, sill bar, glazing
    for x in WIN_X:
        for sill in (SILL_LO, SILL_HI):
            ring = Polygon(arc(x, sill, WIN_R + 0.16, 0, math.pi)
                           + arc(x, sill, WIN_R - 0.14, math.pi, 0))
            r = extrude_polygon(ring, 0.10)
            r.apply_transform(concatenate_matrices(xf, T4(0, 0, T - 0.10)))
            add("trim", r, name)

            g = extrude_polygon(Polygon(
                [(x + WIN_R - 0.10, sill)] + arc(x, sill, WIN_R - 0.10, 0, math.pi)
                + [(x - WIN_R + 0.10, sill)]), 0.08)
            g.apply_transform(concatenate_matrices(xf, T4(0, 0, T - 0.34)))
            add("glass", g, name)

            bar = bx(WIN_R * 2 + 0.32, 0.26, 0.36, x, sill - 0.10, T - 0.10)
            bar.apply_transform(xf)
            add("trim", bar, name)

    if has_door:
        leaf = bx(DOOR_W - 0.25, DOOR_H - 0.1, 0.26, 0, DOOR_H / 2, T - 0.55)
        leaf.apply_transform(xf)
        add("dark", leaf, name)
        lint = bx(DOOR_W + 0.8, 0.4, 0.42, 0, DOOR_H + 0.3, T - 0.10)
        lint.apply_transform(xf)
        add("trim", lint, name)

# ==========================================================================
# PLINTH, CORNICE, CHINESE RAILING  (built once, rotated onto each face)
# ==========================================================================
def face_pieces():
    """Returns (material, mesh) built for the +Z face."""
    out = []
    out.append(("stone", bx(44.9, 1.8, 0.5, 0, -0.1, HALF + 0.24)))

    L = 44 + 2 * EAVE
    y = Y_TOP
    for h, p in ((0.55, 0.45), (0.75, 1.05), (0.87, 1.4)):
        out.append(("trim", bx(L, h, 0.95, 0, y + h / 2, HALF - 0.45 + p)))
        y += h
    out.append(("trim", bx(L, 0.35, 2.3, 0, Y_TOP + CORN_H + 0.17, HALF + 0.3)))

    piers, pw = 5, 2.0
    span = (L - piers * pw) / (piers - 1)
    yb = Y_TOP + CORN_H + 0.35
    for i in range(piers):
        x = -L / 2 + pw / 2 + i * (pw + span)
        out.append(("trim", bx(pw, PARA_H, 1.5, x, yb + PARA_H / 2, HALF + 0.3)))
        if i < piers - 1:
            cx = x + (pw + span) / 2
            out.append(("trim", bx(span, 0.30, 0.7, cx, yb + PARA_H - 0.15, HALF + 0.3)))
            out.append(("trim", bx(span, 0.28, 0.7, cx, yb + 0.5, HALF + 0.3)))
            diag = math.hypot(span, PARA_H - 0.8)
            ang = math.atan2(PARA_H - 0.8, span)
            for s in (1, -1):
                out.append(("trim", bx(diag, 0.22, 0.34, cx,
                                       yb + 0.5 + (PARA_H - 0.8) / 2, HALF + 0.3,
                                       rot=RZ(ang * s))))
    return out


for k, (name, _, _) in enumerate(SIDES):
    # front, rear, left, right  ->  0, 180, 90, -90 degrees
    ang = {"front": 0.0, "rear": math.pi, "left": math.pi / 2, "right": -math.pi / 2}[name]
    for mat, mesh in face_pieces():
        m = mesh.copy()
        m.apply_transform(RY(ang))
        add(mat, m, name)

# ==========================================================================
# ROOF — ridge and furrow with a glazed centre bay, per the section, sheet 7
# ==========================================================================
BAYS, RH = 9, 2.4
bw = 41.0 / BAYS
for i in range(BAYS):
    cx = -20.5 + bw / 2 + i * bw
    prism = extrude_polygon(Polygon([(-bw / 2, 0), (bw / 2, 0), (0, RH)]), 41.0)
    prism.apply_transform(T4(cx, Y_TOP, -20.5))
    add("glass" if i == BAYS // 2 else "roof", prism, "roof")
add("roof", bx(41, 0.5, 41, 0, Y_TOP - 0.25, 0), "roof")

# closed floors for the sealed exterior model, so the shell is not hollow
add("wood", bx(41, 0.5, 41, 0, Y_MUS - 0.25, 0), "core")
add("wood", bx(41, 0.5, 41, 0, Y_TH - 0.25, 0), "core")

# ==========================================================================
# INTERIOR — museum floor, octagonal theatre, centre stair
# ==========================================================================
def flat(poly, thick, y_top):
    """Extrude a plan-view polygon (given in world x,z) downward from y_top."""
    p = Polygon([(x, -z) for x, z in poly.exterior.coords],
                [[(x, -z) for x, z in h.coords] for h in poly.interiors])
    m = extrude_polygon(p, thick)
    m.apply_transform(concatenate_matrices(T4(0, y_top, 0), rotation_matrix(-math.pi / 2, [1, 0, 0])))
    return m


def oct_pts(a):
    R = a / math.cos(math.pi / 8)
    return [(R * math.cos(math.pi / 8 + k * math.pi / 4),
             R * math.sin(math.pi / 8 + k * math.pi / 4)) for k in range(8)]


sq = [(-IH, -IH), (IH, -IH), (IH, IH), (-IH, IH)]
add("wood", flat(Polygon(sq), 1.0, Y_MUS), "inner")

x0, x1, z0, z1 = WELL
add("wood", flat(Polygon(sq, [[(x0, z0), (x1, z0), (x1, z1), (x0, z1)]]), 1.0, Y_TH), "inner")

for i in range(TIERS):
    ai, ao = PIT + i * RUN, PIT + (i + 1) * RUN
    top = Y_TH + (i + 1) * RISE
    Vi, Vo = oct_pts(ai), oct_pts(ao)
    for k in range(8):
        if k == 1 and ao <= z1:      # opened up for the stair well
            continue
        quad = [Vo[k], Vo[(k + 1) % 8], Vi[(k + 1) % 8], Vi[k]]
        add("wood", flat(Polygon(quad), top - Y_TH, top), "inner")

add("wood", flat(Polygon(oct_pts(PIT)), 0.4, Y_TH + 0.05), "inner")
add("trim", bx(5.6, 0.35, 2.6, 0, Y_TH + 3.0, 0), "inner")          # dissecting table
for x, z in ((-2.4, -0.9), (2.4, -0.9), (-2.4, 0.9), (2.4, 0.9)):
    add("trim", bx(0.22, 2.9, 0.22, x, Y_TH + 1.45, z), "inner")
for x, z in ((-6.25, -7.375), (6.25, -7.375), (-6.25, 7.375), (6.25, 7.375)):
    add("trim", bx(0.5, 9.83, 0.5, x, 4.915, z), "inner")           # 6"x6" columns

rs = Y_TH / 16
for i in range(1, 17):
    add("wood", bx(3.83, rs, 0.9, 0, rs * i - rs / 2, 14.9 - (i - 1) * 0.78), "inner")

# rail around the top tier
Vt = oct_pts(PIT + TIERS * RUN)
topY = Y_TH + TIERS * RISE
for k in range(8):
    a, b = Vt[k], Vt[(k + 1) % 8]
    ln = math.hypot(b[0] - a[0], b[1] - a[1])
    ang = -math.atan2(b[1] - a[1], b[0] - a[0])
    add("wood", bx(ln, 0.16, 0.16, (a[0] + b[0]) / 2, topY + 3.0, (a[1] + b[1]) / 2, rot=RY(ang)), "inner")
    add("wood", bx(0.16, 3.0, 0.16, a[0], topY + 1.5, a[1]), "inner")


# ==========================================================================
# ASSEMBLE + EXPORT
# ==========================================================================
def build(sides, path, label):
    scene = trimesh.Scene()
    tris = 0
    for (mat, side), meshes in BUCKETS.items():
        if side not in sides:
            continue
        m = trimesh.util.concatenate([x.copy() for x in meshes])
        m.apply_translation([0, -GROUND, 0])        # ground plane to y = 0
        m.apply_scale(FT)                           # feet -> metres
        col = PAL[mat]
        m.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                name=mat,
                baseColorFactor=[int(c * 255) for c in col] + [140 if mat == "glass" else 255],
                metallicFactor=0.0,
                roughnessFactor=0.35 if mat == "glass" else 0.85,
                alphaMode="BLEND" if mat == "glass" else "OPAQUE",
                doubleSided=True))
        scene.add_geometry(m, geom_name=f"{mat}_{side}")
        tris += len(m.faces)
    scene.export(path)
    b = scene.bounds
    print(f"{label:9s} {path:22s} tris={tris:6d}  "
          f"size = {b[1][0]-b[0][0]:.2f} x {b[1][2]-b[0][2]:.2f} x {b[1][1]-b[0][1]:.2f} m")
    return scene


ALL = {"front", "rear", "left", "right", "core", "roof"}
build(ALL, "/home/claude/theatre.glb", "exterior")
build({"rear", "left", "inner"}, "/home/claude/theatre_cut.glb", "cutaway")
