"""Shared figure style for every CT figure (rule in CLAUDE.md): real and synthetic data look the same.

W400/L40 window, gray, nearest interpolation; organ (label 1) green, tumour (label 2) red, filled at alpha 0.4 with a
0.6 contour on top; volumes reoriented to RAS in memory and shown in radiological convention (patient right on image
left, anterior / superior up), with the aspect taken from the voxel spacing.

    image, label, spacing = to_ras(image, label, affine)
    img2d, lab2d, aspect = plane(image, label, spacing, "axial", k)
    draw(ax, img2d, lab2d, aspect)
"""

import numpy as np

from orient import reorient_array, spacing_of

WINDOW = (40 - 200, 40 + 200)  # W400 / L40
COLORS = {1: (0.0, 0.85, 0.0), 2: (1.0, 0.1, 0.1)}  # organ green, tumour red
ALPHA = 0.4
LINEWIDTH = 0.6
CAPTION = "W400/L40, radiological (patient right = image left). Green organ, red tumour."

# plane -> (axis sliced through, axis shown as rows, axis shown as columns) of a RAS (x, y, z) array
PLANES = {"axial": (2, 1, 0), "coronal": (1, 2, 0), "sagittal": (0, 2, 1)}


def to_ras(image, label, affine):
    """Reorient image + label arrays to RAS; returns (image, label, (sx, sy, sz) in mm)."""
    image, ras_affine = reorient_array(image, affine)
    if label is not None:
        label, _ = reorient_array(label, affine)
    return image, label, spacing_of(ras_affine)


def display(slice2d):
    """RAS slice (columns axis, rows axis) -> image rows anterior/superior first, columns patient right first.
    Works for all three planes: axial (x, y), coronal (x, z), sagittal (y, z; anterior on the left)."""
    return slice2d.T[::-1, ::-1]


def plane(image, label, spacing, name, index):
    """One slice of a RAS volume in display orientation: (image2d, label2d, aspect for imshow)."""
    axis, rows, cols = PLANES[name]
    img = display(np.take(image, index, axis=axis))
    lab = display(np.take(label, index, axis=axis)) if label is not None else None
    return img, lab, spacing[rows] / spacing[cols]


def draw(ax, image, label, aspect=1.0):
    """Windowed CT slice with the label filled and outlined. Arrays must already be in display orientation."""
    ax.imshow(image, cmap="gray", vmin=WINDOW[0], vmax=WINDOW[1], aspect=aspect, interpolation="nearest")
    if label is not None:
        rgba = np.zeros(label.shape + (4,), np.float32)
        for v, c in COLORS.items():
            rgba[label == v] = (*c, ALPHA)
        ax.imshow(rgba, aspect=aspect, interpolation="nearest")
        for v, c in COLORS.items():
            if (label == v).any():
                ax.contour(label == v, levels=[0.5], colors=[c], linewidths=LINEWIDTH)
    ax.set_xticks([]), ax.set_yticks([])


def legend_handles(organ="organ", tumour="tumour"):
    from matplotlib.patches import Patch

    return [Patch(color=COLORS[1], label=organ), Patch(color=COLORS[2], label=tumour)]
