"""
Seamless Flemish bond brick, generated rather than photographed so it can be
recoloured to match the surviving Jefferson brickwork.

Flemish bond -- long stretchers alternating with short header ends in every
course, headers centred over the stretcher below -- is the bond used on the
Academical Village, and its regular vertical rhythm is the most recognisable
signature of the period's brickwork.  Header ends fired closer to the heat and
generally burnt darker, which gives a real wall its faint checkering.

One tile covers 4'-0" wide x 3'-6" high: 16 courses at 2 5/8", four
stretcher+header units at 12".

Writes brick_diffuse.png and brick_normal.png beside this script.
"""
import os
import numpy as np
from PIL import Image, ImageFilter

RES = 512
COURSES = 16
UNITS = 4                 # stretcher + header pairs across the tile
TILE_U_FT = 4.0           # tile width  in feet
TILE_V_FT = 3.5           # tile height in feet

# --- the three colours worth tuning ---------------------------------------
STRETCHER = np.array([152,  70,  48], float)   # face brick, burnt oxide red
HEADER    = np.array([116,  54,  46], float)   # burnt darker, slightly plum
MORTAR    = np.array([163, 151, 132], float)   # warm lime grey, close in value
JITTER    = 0.055         # brick-to-brick variation; period brick is uniform
JOINT     = 3             # joint thickness in pixels


def build():
    h = RES // COURSES                 # 32 px course
    unit = RES // UNITS                # 128 px stretcher + header
    stretch = 84                       # 8" of the 12" unit
    rng = np.random.default_rng(5)

    diffuse = np.zeros((RES, RES, 3), float); diffuse[:] = MORTAR
    height = np.zeros((RES, RES), float)

    for c in range(COURSES):
        y0 = c * h
        off = unit // 2 if c % 2 else 0      # headers land over stretchers
        x, header = -off, False
        while x < RES + unit:
            w = (unit - stretch) if header else stretch
            base = HEADER if header else STRETCHER
            col = np.clip(base * rng.normal(1.0, JITTER) + rng.normal(0, 3.5, 3), 0, 255)
            rows = h - JOINT
            grad = np.linspace(1.035, 0.955, rows)[:, None, None]
            for xa, xb in spans(x + JOINT, x + w, RES):
                diffuse[y0 + JOINT:y0 + h, xa:xb] = np.clip(
                    np.ones((rows, xb - xa, 3)) * col * grad, 0, 255)
                height[y0 + JOINT:y0 + h, xa:xb] = 1.0
            x += w; header = not header

    # low-frequency blotching spanning several tiles, so the repeat disappears
    low = rng.normal(1.0, 0.05, (8, 8))
    low = np.asarray(Image.fromarray(np.clip(low, .82, 1.18).astype(np.float32), mode="F")
                     .resize((RES, RES), Image.BICUBIC), float)[..., None]
    diffuse = np.clip(diffuse * low, 0, 255)

    d = np.asarray(Image.fromarray(diffuse.astype(np.uint8))
                   .filter(ImageFilter.GaussianBlur(0.45)), float)
    hm = np.asarray(Image.fromarray((height * 255).astype(np.uint8))
                    .filter(ImageFilter.GaussianBlur(1.1)), float) / 255.0

    here = os.path.dirname(os.path.abspath(__file__))
    Image.fromarray(d.astype(np.uint8)).save(os.path.join(here, "brick_diffuse.png"))
    Image.fromarray(normal_from(hm, 2.4)).save(os.path.join(here, "brick_normal.png"))
    print(f"Flemish bond, {RES}px, one tile = {TILE_U_FT}' x {TILE_V_FT}' "
          f"= {COURSES} courses")


def spans(a, b, n):
    """Split an x-range so bricks crossing the tile edge wrap seamlessly."""
    out = []
    if a < 0:
        out.append((n + a, n)); a = 0
    if b > n:
        out += [(a, n), (0, b - n)]
    else:
        out.append((a, b))
    return [(int(x), int(y)) for x, y in out if y > x and 0 <= x < n]


def normal_from(hm, strength=2.0):
    gx = np.roll(hm, -1, 1) - np.roll(hm, 1, 1)
    gy = np.roll(hm, -1, 0) - np.roll(hm, 1, 0)
    nx, ny, nz = -gx * strength, -gy * strength, np.ones_like(hm)
    ln = np.sqrt(nx**2 + ny**2 + nz**2)
    return (np.stack([nx/ln, ny/ln, nz/ln], -1) * 0.5 + 0.5).clip(0, 1).__mul__(255).astype(np.uint8)


if __name__ == "__main__":
    build()
