"""
Seamless running-bond brick texture, generated rather than photographed
so it can be recoloured to match the surviving Jefferson brickwork.

One tile covers TILE_FT square: 16 courses of 8" brick with 3/8" joints,
which is the coursing Jefferson's estimate of 4,000 bricks per foot of
height implies for a wall this size.

Writes brick_diffuse.png and brick_normal.png beside this script.
"""
import os
import numpy as np
from PIL import Image, ImageFilter

RES = 512
COURSES = 16          # brick courses per tile
PER_COURSE = 5        # stretchers per tile
TILE_FT = 3.5         # one tile = 3'-6" square

# base brick colour; the model tints this per surface via baseColorFactor
BRICK = np.array([176, 96, 72], float)
MORTAR = np.array([196, 188, 172], float)


def build():
    h = RES // COURSES                      # course height in pixels
    w = RES // PER_COURSE                   # brick length in pixels
    joint = max(2, int(RES * 0.006))        # mortar joint thickness

    rng = np.random.default_rng(11)
    diffuse = np.zeros((RES, RES, 3), float)
    height = np.zeros((RES, RES), float)
    diffuse[:] = MORTAR
    height[:] = 0.0                          # mortar sits back

    for c in range(COURSES):
        y0 = c * h
        offset = (w // 2) if c % 2 else 0    # running bond
        for b in range(PER_COURSE + 1):
            x0 = b * w - offset
            # per-brick colour: handmade brick varies a lot
            tint = BRICK * rng.uniform(0.80, 1.14) + rng.normal(0, 6, 3)
            face = np.clip(tint, 0, 255)
            ys = slice(y0 + joint, y0 + h)
            xs_start, xs_end = x0 + joint, x0 + w
            for xa, xb in wrap_span(xs_start, xs_end, RES):
                # a little within-brick mottling
                rows = ys.stop - ys.start
                grad = np.linspace(1.05, 0.94, rows)[:, None, None]
                blk = np.ones((rows, xb - xa, 3)) * face * grad
                diffuse[ys, xa:xb] = np.clip(blk, 0, 255)
                height[ys, xa:xb] = 1.0

    # soften joints, then a light overall grain
    d = Image.fromarray(diffuse.astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
    diffuse = np.asarray(d, float)


    hm = np.asarray(Image.fromarray((height * 255).astype(np.uint8))
                    .filter(ImageFilter.GaussianBlur(1.2)), float) / 255.0
    normal = height_to_normal(hm, strength=2.6)

    here = os.path.dirname(os.path.abspath(__file__))
    Image.fromarray(diffuse.astype(np.uint8)).save(os.path.join(here, "brick_diffuse.png"))
    Image.fromarray(normal).save(os.path.join(here, "brick_normal.png"))
    print("brick_diffuse.png / brick_normal.png  "
          f"{RES}px, one tile = {TILE_FT}'-0\" = {COURSES} courses")


def wrap_span(a, b, n):
    """Split an x-range so bricks crossing the tile edge wrap seamlessly."""
    a, b = max(a, -n), b
    out = []
    if a < 0:
        out.append((n + a, n))
        a = 0
    if b > n:
        out.append((a, n))
        out.append((0, b - n))
    else:
        out.append((a, b))
    return [(int(x), int(y)) for x, y in out if y > x]


def height_to_normal(hm, strength=2.0):
    gx = np.roll(hm, -1, axis=1) - np.roll(hm, 1, axis=1)
    gy = np.roll(hm, -1, axis=0) - np.roll(hm, 1, axis=0)
    nx, ny, nz = -gx * strength, -gy * strength, np.ones_like(hm)
    ln = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    return (np.stack([nx / ln, ny / ln, nz / ln], -1) * 0.5 + 0.5).clip(0, 1).__mul__(255).astype(np.uint8)


if __name__ == "__main__":
    build()
