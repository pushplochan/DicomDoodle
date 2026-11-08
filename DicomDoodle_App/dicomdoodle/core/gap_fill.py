"""
Closing small gaps in annotation boundaries before they are saved.

This is the original DicomDoodle algorithm (``utils.fill_discontinuities``),
kept line-for-line equivalent so newly saved ``_seg.dcm`` files are identical
to those written by the original tool:

  1. find boundary end-points (pixels with exactly one 8-connected neighbour),
  2. join each end-point to its nearest unused end-point if the gap is at
     most 20 px, drawing a straight line,
  3. thin the result (Zhang-Suen) so the new joins are one pixel wide,
  4. keep every original pixel's label; new join pixels take the label of
     the nearest end-point.
"""
import numpy as np
from scipy.spatial.distance import cdist

MAX_GAP_PX = 20


def get_neighbors(y, x, image):
    """The 8 neighbours of image[y, x], clockwise starting north (P2..P9)."""
    return np.array([
        image[y - 1, x], image[y - 1, x + 1], image[y, x + 1], image[y + 1, x + 1],
        image[y + 1, x], image[y + 1, x - 1], image[y, x - 1], image[y - 1, x - 1],
    ])


def get_transitions(p):
    """Number of 0 -> 1 transitions around the neighbour ring."""
    return np.sum((p[:-1] == 0) & (p[1:] == 1)) + int((p[-1] == 0) & (p[0] == 1))


def thin(binary_image):
    """Zhang-Suen thinning to a one-pixel-wide skeleton."""
    image = binary_image.copy().astype(int)
    while True:
        prev = image.copy()
        # Sub-iteration 1
        to_delete = []
        rows, cols = image.shape
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                if image[y, x] == 0:
                    continue
                p = get_neighbors(y, x, image)
                n = np.sum(p)
                t = get_transitions(p)
                if 2 <= n <= 6 and t == 1 and (p[0] * p[2] * p[4] == 0) and (p[2] * p[4] * p[6] == 0):
                    to_delete.append((y, x))
        for y, x in to_delete:
            image[y, x] = 0
        # Sub-iteration 2
        to_delete = []
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                if image[y, x] == 0:
                    continue
                p = get_neighbors(y, x, image)
                n = np.sum(p)
                t = get_transitions(p)
                if 2 <= n <= 6 and t == 1 and (p[0] * p[2] * p[6] == 0) and (p[0] * p[4] * p[6] == 0):
                    to_delete.append((y, x))
        for y, x in to_delete:
            image[y, x] = 0
        if np.all(prev == image):
            break
    return image


def draw_line(img, p1, p2):
    """Integer points of a straight line from p1 to p2 (both (y, x))."""
    y1, x1 = p1
    y2, x2 = p2
    points = []
    dx = x2 - x1
    dy = y2 - y1
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        return points
    x_inc = dx / steps
    y_inc = dy / steps
    x, y = x1, y1
    for _ in range(steps + 1):
        points.append((int(round(y)), int(round(x))))
        x += x_inc
        y += y_inc
    return points


def fill_discontinuities(img):
    """Join boundary gaps of up to MAX_GAP_PX pixels; see module docstring."""
    binary = (img > 0).astype(int)

    endpoints = []
    rows, cols = binary.shape
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            if binary[y, x] == 1 and np.sum(get_neighbors(y, x, binary)) == 1:
                endpoints.append((y, x))

    if len(endpoints) < 2:
        return img  # nothing to join

    endpoints = np.array(endpoints)
    distances = cdist(endpoints, endpoints)
    connection_mask = np.zeros_like(binary)
    used = set()
    endpoint_intensities = {}

    for i, (y1, x1) in enumerate(endpoints):
        if i in used:
            continue
        valid_distances = distances[i].copy()
        valid_distances[list(used) + [i]] = np.inf
        j = np.argmin(valid_distances)
        if valid_distances[j] > MAX_GAP_PX:
            continue
        y2, x2 = endpoints[j]
        for y, x in draw_line(binary, (y1, x1), (y2, x2)):
            if 0 <= y < rows and 0 <= x < cols:
                connection_mask[y, x] = 1
        used.add(i)
        used.add(j)
        endpoint_intensities[(y1, x1)] = img[y1, x1]
        endpoint_intensities[(y2, x2)] = img[y2, x2]

    combined = np.maximum(binary, connection_mask)
    thinned = thin(combined)

    output = np.zeros_like(img)
    output[thinned > 0] = np.mean(img[img > 0])   # fallback, overwritten below
    output[binary > 0] = img[binary > 0]          # original pixels keep their label

    new_pixels = (thinned > 0) & (binary == 0)
    if np.any(new_pixels):
        new_coords = np.where(new_pixels)
        new_coords = np.array(list(zip(new_coords[0], new_coords[1])))
        endpoint_coords = np.array(list(endpoint_intensities.keys()))
        distances = cdist(new_coords, endpoint_coords)
        nearest_indices = np.argmin(distances, axis=1)
        for idx, (y, x) in enumerate(new_coords):
            nearest_endpoint = endpoint_coords[nearest_indices[idx]]
            output[y, x] = endpoint_intensities[tuple(nearest_endpoint)]

    return output
