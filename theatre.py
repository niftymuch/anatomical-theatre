"""
Anatomical Theatre, University of Virginia (Thomas Jefferson, 1826-1939).
Reconstructed from the Historic American Buildings Survey sheets 1-7.
Authored in feet, exported in metres.  +Z is the front elevation.

States
  1827  as built.  Ridge-and-furrow roof with centre skylight, Chinese railing.
  1837  roof replaced with a hipped roof and cupola.  CUPOLA FORM CONJECTURAL.
  1888  restoration after the 1886 fire: cupola and railing omitted for cost,
        a door and some windows bricked up.  WHICH OPENINGS IS CONJECTURAL.

Outputs
  theatre.glb / .usdz   1827 exterior, flat base, used for AR
  theatre_1837.glb      web only
  theatre_1888.glb      web only
  theatre_cut.glb       1827 cutaway on its hillside, charnel exposed, web only
"""
import math, os
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
import trimesh
from trimesh.creation import extrude_polygon, box, cone, cylinder
from trimesh.transformations import rotation_matrix, translation_matrix, concatenate_matrices
from shapely.geometry import Polygon

np.random.seed(7)
FT = 0.3048

HALF, T, IH = 22.0, 1.5, 20.5
Y_MUS, Y_TH, Y_TOP = 0.0, 10.83, 23.5
CORN_H, PARA_H, EAVE = 2.17, 3.17, 1.4
WIN_R, SILL_LO, SILL_HI = 2.0, 7.5, 19.0
WIN_X = [-15.0, -6.0, 6.0, 15.0]
DOOR_W, DOOR_H = 4.0, 7.5
PIT, RUN, RISE, TIERS = 3.5, 3.0, 1.3333, 5
WELL = (-2.2, 2.2, 3.5, 15.5)

GRADE_F, GRADE_R = -1.0, -12.5     # ground falls ~11'-6" front to rear
Y_CHARNEL, Y_FOOT = -13.5, -15.5   # charnel floor 13'-6" below the museum
GROUND = -1.0                       # datum: front grade becomes y = 0

PAL = {
    "brick": (0.658, 0.337, 0.247),
    "patch": (0.600, 0.330, 0.262),
    "trim":  (0.898, 0.871, 0.804),
    "stone": (0.612, 0.596, 0.541),
    "roof":  (0.490, 0.518, 0.471),
    "glass": (0.737, 0.847, 0.902),
    "dark":  (0.290, 0.251, 0.220),
    "wood":  (0.690, 0.541, 0.361),
    "plaster": (0.882, 0.867, 0.831),
    "earth": (0.361, 0.455, 0.263),
}
TONED = {"brick", "patch", "trim", "stone", "wood", "roof", "earth", "plaster"}
TEXTURED = {"brick", "patch"}          # these get the brick image
TILE_U_FT, TILE_V_FT = 4.0, 3.5        # one tile: 4 stretcher+header units
                                       # wide, 16 courses high

_HERE = os.path.dirname(os.path.abspath(__file__))
def _tex(name):
    try:
        return Image.open(os.path.join(_HERE, name)).convert("RGB")
    except FileNotFoundError:
        return None
BRICK_MAP = _tex("brick_diffuse.png")   # run brick_texture.py to regenerate
BRICK_NRM = _tex("brick_normal.png")


def planar_uv(mesh, tile_u, tile_v):
    """Box-project UVs from world position, per face, so brick coursing runs
    horizontally on every wall without unwrapping anything."""
    mesh.unmerge_vertices()                     # each face gets its own vertices
    f = mesh.faces
    n = mesh.face_normals
    dom = np.argmax(np.abs(n), axis=1)[:, None]
    c = mesh.vertices[f]                        # (F, 3, 3)
    u = np.where(dom == 0, c[:, :, 2], c[:, :, 0])
    v = np.where(dom == 1, c[:, :, 2], c[:, :, 1])
    uv = np.zeros((len(mesh.vertices), 2))
    uv[f] = np.stack([u / tile_u, v / tile_v], axis=-1)
    return uv


def ambient(mesh, radius=1.7, k=56, floor=0.34):
    """Point-based ambient occlusion: how much of each face's hemisphere is
    blocked by nearby geometry.  Darkens window reveals, inside corners and
    the underside of the cornice, which is most of what makes a render read
    as a building rather than a coloured box."""
    C, N, A = mesh.triangles_center, mesh.face_normals, mesh.area_faces
    dist, idx = cKDTree(C).query(C, k=min(k, len(C)))
    d = C[idx] - C[:, None, :]                      # (F, k, 3)
    r2 = np.maximum((d ** 2).sum(-1), 1e-8)
    u = d / np.sqrt(r2)[..., None]
    cos_i = np.maximum(0.0, np.einsum("fkj,fj->fk", u, N))
    cos_j = np.maximum(0.0, -np.einsum("fkj,fkj->fk", u, N[idx]))
    w = A[idx] * cos_i * cos_j / (np.pi * r2 + A[idx])
    w[dist > radius] = 0.0
    w[:, 0] = 0.0                                   # the face itself
    # a single value per face is only meaningful on small faces; a wall
    # triangle spans metres, so damp occlusion out as area grows
    damp = np.clip(0.30 / np.maximum(A, 1e-9), 0.0, 1.0)
    return np.clip(1.0 - w.sum(1) * damp, floor, 1.0)


def T4(x=0.0, y=0.0, z=0.0): return translation_matrix([x, y, z])
def RY(a): return rotation_matrix(a, [0, 1, 0])
def RZ(a): return rotation_matrix(a, [0, 0, 1])
def RX(a): return rotation_matrix(a, [1, 0, 0])


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


def lune(x, sill, r=WIN_R):
    return [(x + r, sill)] + arc(x, sill, r, 0, math.pi) + [(x - r, sill)]


def flat(poly, thick, y_top):
    """Extrude a plan polygon (world x,z) downward from y_top."""
    p = Polygon([(x, -z) for x, z in poly.exterior.coords],
                [[(x, -z) for x, z in h.coords] for h in poly.interiors])
    m = extrude_polygon(p, thick)
    m.apply_transform(concatenate_matrices(T4(0, y_top, 0), RX(-math.pi / 2)))
    return m


def oct_pts(a):
    R = a / math.cos(math.pi / 8)
    return [(R * math.cos(math.pi / 8 + k * math.pi / 4),
             R * math.sin(math.pi / 8 + k * math.pi / 4)) for k in range(8)]


# ============================================================== build
def build(state="1827", cut=False, hill=False):
    B = []
    def add(mat, side, mesh): B.append((mat, side, mesh))

    railing = state in ("1827", "1837")
    ridge_furrow = state == "1827"
    cupola = state == "1837"
    blocked = state == "1888"
    y_bot = Y_FOOT if hill else GROUND

    sides = [
        ("front", True,  T4(0, 0, HALF - T)),
        ("rear",  True,  concatenate_matrices(T4(0, 0, -(HALF - T)), RY(math.pi))),
        ("left",  False, concatenate_matrices(T4(-HALF, 0, 0), RY(math.pi / 2))),
        ("right", False, concatenate_matrices(T4(HALF, 0, 0), RY(-math.pi / 2))),
    ]
    for name, has_door, xf in sides:
        skip, door = set(), has_door
        if blocked and name == "rear":
            door = False
            skip = {(-6.0, SILL_HI), (6.0, SILL_HI)}

        holes = [lune(x, s) for x in WIN_X for s in (SILL_LO, SILL_HI) if (x, s) not in skip]
        if door:
            holes.append([(-DOOR_W / 2, Y_MUS), (DOOR_W / 2, Y_MUS),
                          (DOOR_W / 2, Y_MUS + DOOR_H), (-DOOR_W / 2, Y_MUS + DOOR_H)])
        w = extrude_polygon(
            Polygon([(-HALF, y_bot), (HALF, y_bot), (HALF, Y_TOP), (-HALF, Y_TOP)], holes), T)
        w.apply_transform(xf)
        # the inward face of a brick wall was plastered and whitewashed, as the
        # 1930s museum model of the interior shows
        out_dir = np.array({"front": (0, 0, 1), "rear": (0, 0, -1),
                            "left": (-1, 0, 0), "right": (1, 0, 0)}[name], float)
        inward = w.face_normals @ out_dir < -0.7
        if inward.any() and (~inward).any():
            add("plaster", name, w.submesh([np.where(inward)[0]], repair=False)[0])
            add("brick", name, w.submesh([np.where(~inward)[0]], repair=False)[0])
        else:
            add("brick", name, w)

        if blocked and name == "rear":
            for x, s in skip:
                p = extrude_polygon(Polygon(lune(x, s)), 0.16)
                p.apply_transform(concatenate_matrices(xf, T4(0, 0, T - 0.16)))
                add("patch", name, p)
            d = bx(DOOR_W, DOOR_H, 0.16, 0, DOOR_H / 2, T - 0.16)
            d.apply_transform(xf); add("patch", name, d)

        for x in WIN_X:
            for s in (SILL_LO, SILL_HI):
                if (x, s) in skip:
                    continue
                ring = Polygon(arc(x, s, WIN_R + 0.16, 0, math.pi)
                               + arc(x, s, WIN_R - 0.14, math.pi, 0))
                r = extrude_polygon(ring, 0.10)
                r.apply_transform(concatenate_matrices(xf, T4(0, 0, T - 0.10)))
                add("trim", name, r)
                g = extrude_polygon(Polygon(lune(x, s, WIN_R - 0.10)), 0.08)
                g.apply_transform(concatenate_matrices(xf, T4(0, 0, T - 0.34)))
                add("glass", name, g)
                pieces = [bx(WIN_R * 2 + 0.32, 0.26, 0.36, x, s - 0.10, T - 0.10)]
                for k in range(1, 6):                 # five radiating spokes
                    a = math.pi * k / 6
                    ln = WIN_R - 0.18
                    pieces.append(bx(ln, 0.12, 0.16,
                                     x + math.cos(a) * ln / 2,
                                     s + math.sin(a) * ln / 2,
                                     T - 0.28, rot=RZ(a)))
                for m in pieces:
                    m.apply_transform(xf); add("trim", name, m)

        if door:
            leaf = bx(DOOR_W - 0.25, DOOR_H - 0.1, 0.26, 0, DOOR_H / 2, T - 0.55)
            leaf.apply_transform(xf); add("dark", name, leaf)
            lint = bx(DOOR_W + 0.8, 0.4, 0.42, 0, DOOR_H + 0.3, T - 0.10)
            lint.apply_transform(xf); add("trim", name, lint)

    # ---- plinth, cornice with dentils, railing -----------------------
    def face_pieces():
        out = [("stone", bx(44.9, 1.8, 0.5, 0, -0.1, HALF + 0.24))]
        L = 44 + 2 * EAVE
        y = Y_TOP
        for h, p in ((0.55, 0.45), (0.75, 1.05), (0.87, 1.4)):
            out.append(("trim", bx(L, h, 0.95, 0, y + h / 2, HALF - 0.45 + p)))
            y += h
        for i in range(29):
            dx = -L / 2 + 0.9 + i * (L - 1.8) / 28
            out.append(("trim", bx(0.42, 0.42, 0.55, dx, Y_TOP + 0.80, HALF + 0.55)))
        out.append(("trim", bx(L, 0.35, 2.3, 0, Y_TOP + CORN_H + 0.17, HALF + 0.3)))

        if railing:
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
                    dg = math.hypot(span, PARA_H - 0.8)
                    ang = math.atan2(PARA_H - 0.8, span)
                    for sgn in (1, -1):
                        out.append(("trim", bx(dg, 0.22, 0.34, cx,
                                               yb + 0.5 + (PARA_H - 0.8) / 2, HALF + 0.3,
                                               rot=RZ(ang * sgn))))
        else:
            out.append(("trim", bx(L, 1.1, 1.6, 0, Y_TOP + CORN_H + 0.9, HALF + 0.3)))
        return out

    for name in ("front", "rear", "left", "right"):
        ang = {"front": 0.0, "rear": math.pi, "left": math.pi / 2, "right": -math.pi / 2}[name]
        for mat, mesh in face_pieces():
            m = mesh.copy(); m.apply_transform(RY(ang)); add(mat, name, m)

    # ---- roof ---------------------------------------------------------
    if ridge_furrow:
        bays, rh = 9, 2.4
        bw = 41.0 / bays
        for i in range(bays):
            cx = -20.5 + bw / 2 + i * bw
            pr = extrude_polygon(Polygon([(-bw / 2, 0), (bw / 2, 0), (0, rh)]), 41.0)
            pr.apply_transform(T4(cx, Y_TOP, -20.5))
            add("glass" if i == bays // 2 else "roof", "roof", pr)
    else:
        hip = cone(radius=41 / math.sqrt(2) * 1.03, height=6.4, sections=4)
        hip.apply_transform(concatenate_matrices(T4(0, Y_TOP, 0), RX(-math.pi / 2), RZ(math.pi / 4)))
        add("roof", "roof", hip)
        if cupola:
            for mat, r, h, yy in (("trim", 3.4, 4.6, Y_TOP + 5.2),
                                  ("glass", 2.9, 3.2, Y_TOP + 5.4)):
                c = cylinder(radius=r, height=h, sections=8)
                c.apply_transform(concatenate_matrices(T4(0, yy + h / 2, 0), RX(-math.pi / 2)))
                add(mat, "roof", c)
            dm = cone(radius=3.7, height=2.8, sections=8)
            dm.apply_transform(concatenate_matrices(T4(0, Y_TOP + 9.8, 0), RX(-math.pi / 2)))
            add("roof", "roof", dm)
    add("roof", "roof", bx(41, 0.5, 41, 0, Y_TOP - 0.25, 0))

    # ---- sealed floors for the AR shell --------------------------------
    if not cut:
        add("wood", "core", bx(41, 0.5, 41, 0, Y_MUS - 0.25, 0))
        add("wood", "core", bx(41, 0.5, 41, 0, Y_TH - 0.25, 0))

    # ---- interior ------------------------------------------------------
    if cut:
        sq = [(-IH, -IH), (IH, -IH), (IH, IH), (-IH, IH)]
        ch = [[(-18.2, -18.2), (-13.3, -18.2), (-13.3, -13.3), (-18.2, -13.3)]] if hill else []
        add("wood", "inner", flat(Polygon(sq, ch), 1.0, Y_MUS))
        x0, x1, z0, z1 = WELL
        add("wood", "inner",
            flat(Polygon(sq, [[(x0, z0), (x1, z0), (x1, z1), (x0, z1)]]), 1.0, Y_TH))
        for i in range(TIERS):
            ai, ao = PIT + i * RUN, PIT + (i + 1) * RUN
            top = Y_TH + (i + 1) * RISE
            Vi, Vo = oct_pts(ai), oct_pts(ao)
            for k in range(8):
                if k == 1 and ao <= z1:
                    continue
                add("wood", "inner", flat(Polygon(
                    [Vo[k], Vo[(k + 1) % 8], Vi[(k + 1) % 8], Vi[k]]), top - Y_TH, top))
                a, b = Vi[k], Vi[(k + 1) % 8]        # dark cap along the tier edge
                ln = math.hypot(b[0] - a[0], b[1] - a[1])
                add("dark", "inner", bx(ln, 0.30, 0.34, (a[0] + b[0]) / 2, top + 0.15,
                                        (a[1] + b[1]) / 2,
                                        rot=RY(-math.atan2(b[1] - a[1], b[0] - a[0]))))
        add("wood", "inner", flat(Polygon(oct_pts(PIT)), 0.4, Y_TH + 0.05))
        add("trim", "inner", bx(5.6, 0.35, 2.6, 0, Y_TH + 3.0, 0))
        for x, z in ((-2.4, -0.9), (2.4, -0.9), (-2.4, 0.9), (2.4, 0.9)):
            add("trim", "inner", bx(0.22, 2.9, 0.22, x, Y_TH + 1.45, z))
        for x, z in ((-6.25, -7.375), (6.25, -7.375), (-6.25, 7.375), (6.25, 7.375)):
            add("trim", "inner", bx(0.5, 9.83, 0.5, x, 4.915, z))
        rs = Y_TH / 16
        for i in range(1, 17):
            add("wood", "inner", bx(3.83, rs, 0.9, 0, rs * i - rs / 2, 14.9 - (i - 1) * 0.78))
        Vt, topY = oct_pts(PIT + TIERS * RUN), Y_TH + TIERS * RISE
        for k in range(8):
            a, b = Vt[k], Vt[(k + 1) % 8]
            ln = math.hypot(b[0] - a[0], b[1] - a[1])
            add("wood", "inner", bx(ln, 0.16, 0.16, (a[0] + b[0]) / 2, topY + 3.0,
                                    (a[1] + b[1]) / 2,
                                    rot=RY(-math.atan2(b[1] - a[1], b[0] - a[0]))))
            add("wood", "inner", bx(0.16, 3.0, 0.16, a[0], topY + 1.5, a[1]))

    # ---- hillside, foundation and charnel ------------------------------
    if hill:
        add("stone", "base", bx(24, 0.6, 24, -8.5, Y_CHARNEL - 0.3, -8.5))
        add("stone", "base", bx(0.9, 12.5, 11.5, -9, Y_CHARNEL + 6.25, -14.75))
        add("stone", "base", bx(11.5, 12.5, 0.9, -14.75, Y_CHARNEL + 6.25, -9))
        # earth under the rest of the museum floor, shown as a shallow band so
        # the charnel stays visible from the cut side
        add("earth", "base", bx(29.5, 2.0, 41, 5.75, -2.0, 0))
        add("earth", "base", bx(11.5, 2.0, 29.5, -14.75, -2.0, 5.75))
        for i in range(20):                                   # winder to the charnel
            w = cylinder(radius=2.3, height=0.24, sections=6)
            w.apply_transform(concatenate_matrices(
                T4(-15.75, -0.7 - i * (12.6 / 20), -15.75),
                RX(-math.pi / 2), RZ(-i * (math.pi * 2.4 / 20))))
            add("wood", "base", w)
        add("dark", "base", bx(3.4, 6.0, 0.5, -9.0, GRADE_R + 3.0, -HALF + 0.25))

        # terrain only behind the retained faces, so the cutaway stays open
        rear = extrude_polygon(Polygon([(-44, GRADE_R), (44, GRADE_R),
                                        (44, -22), (-44, -22)]), 22.0)
        rear.apply_transform(T4(0, 0, -44))
        add("earth", "site", rear)
        left = extrude_polygon(Polygon([(-HALF - 22, GRADE_F + 0.4), (HALF, GRADE_R),
                                        (HALF, -22), (-HALF - 22, -22)]), 22.0)
        left.apply_transform(concatenate_matrices(T4(-44, 0, 0), RY(math.pi / 2)))
        add("earth", "site", left)

        for i in range(20):                                   # outside stair
            h = -(12.5 / 20) * i - (GRADE_R - 1)
            add("stone", "site", bx(0.85, h, 5.4, 4.2 + i * 0.85,
                                    -(12.5 / 20) * i - h / 2, -HALF - 3.0))
        add("stone", "site", bx(7.5, 1.1, 6.0, 0, -0.55, -HALF - 3.0))

    return B


# ================================================= tonal variation + export
def tone(mesh, ao):
    n, c = mesh.face_normals, mesh.triangles_center
    f = ao.copy()
    f *= np.where(n[:, 1] < -0.3, 0.86, 1.0)          # undersides in shade
    f *= np.where(n[:, 1] > 0.70, 1.04, 1.0)          # tops catch the light
    f *= 0.93 + 0.07 * np.clip(c[:, 1] / 1.8, 0, 1)   # weathering near grade
    return f


def export(B, sides, path, label):
    groups = {}
    for mat, side, mesh in B:
        if side in sides:
            groups.setdefault(mat, []).append(mesh)

    parts = []
    for mat, meshes in groups.items():
        m = trimesh.util.concatenate([x.copy() for x in meshes])
        m.apply_translation([0, -GROUND, 0])
        m.apply_scale(FT)
        parts.append((mat, m))

    world = trimesh.util.concatenate([m for _, m in parts])
    assert len(world.faces) == sum(len(m.faces) for _, m in parts)
    ao_all = ambient(world)

    scene, tris, cursor = trimesh.Scene(), 0, 0
    for mat, m in parts:
        n = len(m.faces)
        ao = ao_all[cursor:cursor + n]; cursor += n
        base = np.array(PAL[mat])
        alpha = 150 if mat == "glass" else 255
        textured = mat in TEXTURED and BRICK_MAP is not None

        def emit(sub, k, name):
            nonlocal tris
            uv = planar_uv(sub, TILE_U_FT * FT, TILE_V_FT * FT) if textured else None
            col = np.clip((np.array([k, k, k]) if textured else base * k), 0, 1)
            kw = dict(name=name,
                      baseColorFactor=[int(v * 255) for v in col] + [alpha],
                      metallicFactor=0.0,
                      roughnessFactor=0.35 if mat == "glass" else 0.88,
                      alphaMode="BLEND" if mat == "glass" else "OPAQUE",
                      doubleSided=True)
            if textured:
                kw["baseColorTexture"] = BRICK_MAP
                if BRICK_NRM is not None:
                    kw["normalTexture"] = BRICK_NRM
            sub.visual = trimesh.visual.TextureVisuals(
                uv=uv, material=trimesh.visual.material.PBRMaterial(**kw))
            scene.add_geometry(sub, geom_name=f"{name}_{len(scene.geometry)}")
            tris += len(sub.faces)

        if mat == "glass":
            emit(m, 1.0, mat)
            continue

        f = tone(m, ao)
        if textured:
            # wall polygons triangulate into long slivers: small area, metres
            # long.  Any per-face shading shows up as dark fans radiating from
            # the openings, so brick is left completely uniform.
            f = np.ones(len(f))
        bins = np.digitize(f, [0.55, 0.72, 0.85, 0.94])
        for b in np.unique(bins):
            i = np.where(bins == b)[0]
            emit(m.submesh([i], repair=False)[0], float(np.mean(f[i])), f"{mat}{b}")

    scene.export(path)
    bb = scene.bounds
    print(f"{label:14s} {path.split('/')[-1]:20s} tris={tris:6d}  "
          f"{bb[1][0]-bb[0][0]:5.2f} x {bb[1][2]-bb[0][2]:5.2f} x {bb[1][1]-bb[0][1]:5.2f} m"
          f"   floor y={bb[0][1]:+.2f}  ao {ao_all.min():.2f}-{ao_all.max():.2f}")


if __name__ == "__main__":
    import os
    OUT = os.path.dirname(os.path.abspath(__file__)) + "/"   # writes beside this script
    SHELL = {"front", "rear", "left", "right", "core", "roof"}
    CUT = {"rear", "left", "inner", "base", "site"}
    export(build("1827"), SHELL, OUT + "theatre.glb", "flat, for AR")
    export(build("1827", hill=True), SHELL | {"base", "site"},
           OUT + "theatre_hill.glb", "on its hill")
    export(build("1827", cut=True, hill=True), CUT, OUT + "theatre_cut.glb", "cutaway")
