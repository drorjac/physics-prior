"""Palette validation.

A colour choice for a chart is checkable, so it is checked rather than
eyeballed. This is a Python port of the data-visualisation method's validator
(the reference implementation is JavaScript and node is not a dependency
here): the same Machado, Oliveira & Fernandes (2009) colour-vision-deficiency
matrices at severity 1.0, the same Euclidean OKLab dE x100, and the same
lightness-band, chroma-floor, CVD-separation, normal-vision and contrast
gates. It reproduces the reference implementation's published numbers exactly
(eight-slot adjacent, light surface: worst CVD dE 9.1, normal-vision 19.6).

    python -m physprior.viz.palette "#2a78d6,#eb6834,#1baf7a,#4a3aa7" light all
"""

from __future__ import annotations

import itertools
import sys

import numpy as np

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
CHROMA_FLOOR, CVD_TARGET, CVD_FLOOR, NORMAL_FLOOR, CONTRAST_MIN = (
    0.10,
    8.0,
    6.0,
    15.0,
    3.0,
)
MACHADO = {
    "protan": [
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998],
    ],
    "deutan": [
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ],
    "tritan": [
        [1.255528, -0.076749, -0.178779],
        [-0.078411, 0.930809, 0.147602],
        [0.004733, 0.691367, 0.303900],
    ],
}


def lin(h):
    h = h.lstrip("#")
    c = np.array([int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def oklab(rgb_lin):
    m = np.array(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ]
    )
    lms = np.cbrt(m @ rgb_lin)
    n = np.array(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ]
    )
    return n @ lms


def sim(h, kind):
    return np.clip(np.array(MACHADO[kind]) @ lin(h), 0, 1)


def dE(a, b, kind=None):
    x = oklab(sim(a, kind) if kind else lin(a))
    y = oklab(sim(b, kind) if kind else lin(b))
    return 100 * float(np.linalg.norm(x - y))


def rel_lum(h):
    r, g, b = lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    x, y = sorted([rel_lum(a), rel_lum(b)], reverse=True)
    return (x + 0.05) / (y + 0.05)


def validate(pal, mode="light", pairs="adjacent"):
    surf = SURFACE[mode]
    lo, hi = BAND[mode]
    rep, ok = [], True
    Ls = [oklab(lin(c))[0] for c in pal]
    Cs = [float(np.hypot(*oklab(lin(c))[1:])) for c in pal]
    off = [(c, round(L, 3)) for c, L in zip(pal, Ls) if lo > L or hi < L]
    rep.append(
        ("Lightness band", "pass" if not off else "warn", off or f"all in [{lo},{hi}]")
    )
    lowc = [(c, round(C, 3)) for c, C in zip(pal, Cs) if C < CHROMA_FLOOR]
    rep.append(("Chroma floor", "pass" if not lowc else "fail", lowc or "all >= 0.10"))
    if lowc:
        ok = False
    idx = (
        list(itertools.combinations(range(len(pal)), 2))
        if pairs == "all"
        else [(i, i + 1) for i in range(len(pal) - 1)]
    )
    worst = min(
        (
            (dE(pal[i], pal[j], k), k, pal[i], pal[j])
            for k in ("protan", "deutan")
            for i, j in idx
        ),
        key=lambda t: t[0],
    )
    tri = min(dE(pal[i], pal[j], "tritan") for i, j in idx)
    st = (
        "pass"
        if worst[0] >= CVD_TARGET
        else "floor"
        if worst[0] >= CVD_FLOOR
        else "fail"
    )
    if st == "fail":
        ok = False
    rep.append(
        (
            "CVD separation",
            st,
            f"worst {pairs} {worst[2]}<->{worst[3]} dE {worst[0]:.1f} ({worst[1]}) - tritan {tri:.1f}",
        )
    )
    nw = min(((dE(pal[i], pal[j]), pal[i], pal[j]) for i, j in idx), key=lambda t: t[0])
    st = "pass" if nw[0] >= NORMAL_FLOOR else "fail"
    if st == "fail":
        ok = False
    rep.append(("Normal-vision floor", st, f"worst {nw[1]}<->{nw[2]} dE {nw[0]:.1f}"))
    low = [
        (c, round(contrast(c, surf), 2))
        for c in pal
        if contrast(c, surf) < CONTRAST_MIN
    ]
    rep.append(
        (
            "Contrast vs surface",
            "relief" if low else "pass",
            low or f"all >= {CONTRAST_MIN}:1",
        )
    )
    return rep, ok


#: The four competing arms, validated all-pairs on the light surface.
ARM_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]


def main(argv: list[str] | None = None) -> int:
    """Print a validation report. Returns 0 when every hard gate passes."""
    args = list(sys.argv[1:] if argv is None else argv)
    pal = [c.strip() for c in args[0].split(",") if c.strip()] if args else ARM_PALETTE
    mode = args[1] if len(args) > 1 else "light"
    pairs = args[2] if len(args) > 2 else "adjacent"
    rep, ok = validate(pal, mode, pairs)
    print(f"palette {pal}  mode={mode}  pairs={pairs}")
    for name, state, detail in rep:
        print(f"  {state.upper():7s} {name:22s} {detail}")
    print("  => OK" if ok else "  => FAIL")
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
