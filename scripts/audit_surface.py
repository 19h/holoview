#!/usr/bin/env python3
"""Check resampled HYPC points against the independently decoded source mesh.

Reference HYPC must contain the CRS-corrected original vertices in OBJ order.
Uses exact point-to-triangle distances (not nearest-vertex distances). A nearby
triangle within the quantization bound suffices; failed centroid candidates are
expanded with a conservative global triangle-radius bound.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from scipy.spatial import cKDTree
from audit_alignment import read_hypc


def read_faces(path, cache):
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / (path.stem + '-faces.npy')
    if out.exists() and out.stat().st_mtime_ns >= path.stat().st_mtime_ns:
        return np.load(out, mmap_mode='r')
    faces = []
    vertices = 0
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith('.obj')]
        assert len(names) == 1
        with z.open(names[0]) as obj:
            for line in obj:
                f = line.split()
                if not f:
                    continue
                if f[0] == b'v':
                    vertices += 1
                elif f[0] == b'f':
                    assert len(f) == 4, 'Triangulated source required'
                    ids = [int(v.split(b'/')[0]) for v in f[1:]]
                    faces.append([v - 1 if v > 0 else vertices + v for v in ids])
    result = np.asarray(faces, dtype=np.int32)
    assert result.size and result.min() >= 0 and result.max() < vertices
    np.save(out, result)
    return result


def distances(points, triangles):
    """Broadcast arrays (...,3), (...,3,3); return triangle distances in metres."""
    a, b, c = triangles[..., 0, :], triangles[..., 1, :], triangles[..., 2, :]
    ab, ac, ap = b-a, c-a, points-a
    dot = lambda x, y: np.sum(x*y, axis=-1)
    aa, cc, abac = dot(ab, ab), dot(ac, ac), dot(ab, ac)
    denom = aa*cc-abac*abac
    safe = np.maximum(denom, 1e-30)
    u = (cc*dot(ap, ab)-abac*dot(ap, ac))/safe
    v = (aa*dot(ap, ac)-abac*dot(ap, ab))/safe
    normal = np.cross(ab, ac)
    plane = dot(ap, normal)**2/np.maximum(dot(normal, normal), 1e-30)
    best = np.where((denom > 1e-24) & (u >= 0) & (v >= 0) & (u+v <= 1), plane, np.inf)
    for start, end in [(a,b),(a,c),(b,c)]:
        edge = end-start
        t = np.clip(dot(points-start, edge)/np.maximum(dot(edge,edge), 1e-30), 0, 1)
        diff = points-(start+t[...,None]*edge)
        best = np.minimum(best, dot(diff,diff))
    return np.sqrt(np.maximum(best, 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--surface', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    results = []
    for path in sorted(args.surface.glob('*.hypc')):
        source = args.source / (path.stem + '.zip')
        faces = read_faces(source, args.cache)
        vertices, ref_upm = read_hypc(args.reference/path.name)
        points, upm = read_hypc(path)
        triangles = vertices[faces]
        centers = triangles.mean(axis=1)
        tree = cKDTree(centers)
        indices = np.unique(np.linspace(0, len(points)-1, min(4096, len(points)), dtype=int))
        samples = points[indices]
        _, candidate = tree.query(samples, k=min(32, len(faces)), workers=-1)
        error = distances(samples[:,None,:], triangles[candidate]).min(axis=1)
        bound = np.sqrt(3)*0.5*(1/ref_upm+1/upm) + 1e-7
        missing = np.flatnonzero(error > bound)
        if len(missing):
            max_radius = np.linalg.norm(triangles-centers[:,None,:], axis=2).max()
            for i in missing:
                candidates = tree.query_ball_point(samples[i], max_radius+bound)
                error[i] = distances(samples[i], triangles[candidates]).min()
        assert error.max() <= bound, (path, error.max(), bound)
        row = {'tile': path.name, 'source_vertices': len(vertices), 'source_faces': len(faces),
               'surface_points': len(points), 'samples': len(samples),
               'max_surface_distance_m': float(error.max()), 'bound_m': bound,
               'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        results.append(row)
        print(json.dumps(row), flush=True)
    assert results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
