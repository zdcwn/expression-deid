"""Local identity proportion edits that preserve within-feature motion ratios.

The mesh is a geometric condition, never an expression-classifier input.
Every eye/brow group and the lips use one affine transform, so eyelid and
mouth aperture ratios and eyebrow-to-eye relationships survive the edit.
The face boundary is anchored; hair and background are outside the warp.
"""
from __future__ import annotations

import numpy as np


def _indices(connections):
    return sorted({value for edge in connections for value in (edge.start, edge.end)})


def identity_geometry(points, connections, strength=1.0, profile=0):
    source = np.asarray(points, dtype=np.float32)
    target = source.copy()
    # Values above one deliberately exaggerate identity proportions while
    # keeping the expression-bearing vertical coordinates unchanged.  The
    # privacy selector uses this bounded extension to generate stronger
    # identity-diverse candidates; it never changes the face oval itself.
    strength = float(np.clip(strength, 0.0, 2.0))
    eye_right = _indices(connections.FACE_LANDMARKS_RIGHT_EYE)
    eye_left = _indices(connections.FACE_LANDMARKS_LEFT_EYE)
    right_center = source[eye_right].mean(axis=0)
    left_center = source[eye_left].mean(axis=0)
    horizontal = left_center - right_center
    eye_distance = float(np.linalg.norm(horizontal))
    if eye_distance < 0.03:
        raise RuntimeError("Face geometry has insufficient eye separation")
    horizontal /= eye_distance
    vertical = np.array([-horizontal[1], horizontal[0]], dtype=np.float32)
    # Image y points downward. Keep the local head coordinate system upright.
    if vertical[1] < 0:
        vertical *= -1
    basis = np.stack([horizontal, vertical], axis=1)

    # All profiles keep the vertical scale and vertical shift at their source
    # values.  Only horizontal proportions move, so eyelid aperture, mouth
    # opening, brow height and expression geometry are not replaced.
    profiles = (
        (0.58, 1.55, 0.55, 0.14),
        (0.30, 1.95, 0.28, 0.27),
        (0.78, 1.28, 0.72, 0.36),
        (0.42, 1.72, 0.42, 0.22),
        (0.88, 2.00, 0.34, 0.30),
        (0.18, 1.45, 0.82, 0.40),
    )
    profile_index = int(np.clip(profile, 0, len(profiles) - 1))
    eye_scale, nose_scale, lip_scale, eye_shift = profiles[profile_index]

    def edit(ids, center, scale_x, scale_y, shift_x, shift_y):
        local = (source[ids] - center) @ basis
        scales = 1.0 + strength * (np.array([scale_x, scale_y]) - 1.0)
        offset = eye_distance * strength * np.array([shift_x, shift_y])
        target[ids] = (local * scales + offset) @ basis.T + center

    right = sorted(set(eye_right + _indices(connections.FACE_LANDMARKS_RIGHT_EYEBROW) + _indices(connections.FACE_LANDMARKS_RIGHT_IRIS)))
    left = sorted(set(eye_left + _indices(connections.FACE_LANDMARKS_LEFT_EYEBROW) + _indices(connections.FACE_LANDMARKS_LEFT_IRIS)))
    # Keep eyelid aperture and brow height unchanged: the vertical geometry is
    # an expression cue, while eye spacing/width still supplies an identity
    # edit.
    edit(right, right_center, eye_scale, 1.0, -eye_shift, 0.0)
    edit(left, left_center, eye_scale, 1.0, eye_shift, 0.0)
    nose = _indices(connections.FACE_LANDMARKS_NOSE)
    # Change nasal width without moving the nose vertically.
    edit(nose, source[1], nose_scale, 1.0, 0.0, 0.0)
    lips = _indices(connections.FACE_LANDMARKS_LIPS)
    # Preserve mouth opening and corner height; only alter horizontal lip
    # proportions so a neutral/smile/frown cannot be turned into another
    # expression by the anonymizer geometry.
    edit(lips, source[lips].mean(axis=0), lip_scale, 1.0, 0.0, 0.0)
    # Explicitly pin face outline even if a connection set ever overlaps.
    outline = _indices(connections.FACE_LANDMARKS_FACE_OVAL)
    target[outline] = source[outline]
    return target


def warp_features(gray, source_points, target_points):
    """Piecewise affine warp with the boundary and exterior exactly pinned."""
    import cv2
    from scipy.spatial import Delaunay

    image = np.asarray(gray, dtype=np.uint8)
    height, width = image.shape
    scale = np.array([width - 1, height - 1], dtype=np.float64)
    source = np.asarray(source_points, dtype=np.float64) * scale
    target = np.asarray(target_points, dtype=np.float64) * scale
    corners = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], dtype=np.float64)
    source = np.concatenate([source, corners])
    target = np.concatenate([target, corners])
    triangles = Delaunay(target)
    yy, xx = np.indices(image.shape, dtype=np.float32)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    cells = triangles.find_simplex(grid)
    valid = cells >= 0
    transforms = triangles.transform[cells[valid]]
    relative = grid[valid] - transforms[:, 2, :]
    barycentric = np.einsum("nij,nj->ni", transforms[:, :2, :], relative)
    barycentric = np.column_stack([barycentric, 1.0 - barycentric.sum(axis=1)])
    vertices = source[triangles.simplices[cells[valid]]]
    mapped = np.einsum("ni,nij->nj", barycentric, vertices)
    map_x, map_y = xx.ravel().copy(), yy.ravel().copy()
    map_x[valid], map_y[valid] = mapped[:, 0], mapped[:, 1]
    return cv2.remap(image, map_x.reshape(image.shape), map_y.reshape(image.shape), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
