"""Marker shapes by sign so the figures also read without colour: excitatory = circle, inhibitory = triangle, unknown = diamond."""
BLUE, ORANGE = "#2a78d6", "#eb6834"
MARK = {BLUE: ("o", 1.0), ORANGE: ("^", 1.35), "#a8a7a2": ("D", 0.8)}


def scatter_by_color(ax, x, y, s, colors, **kw):
    import numpy as np
    x, y = np.asarray(x), np.asarray(y)
    s = np.broadcast_to(np.asarray(s, dtype=float), x.shape)
    colors = list(colors)
    for c, (m, k) in MARK.items():
        idx = [i for i, cc in enumerate(colors) if cc == c]
        if idx:
            ax.scatter(x[idx], y[idx], s=s[idx] * k, c=c, marker=m, **kw)
