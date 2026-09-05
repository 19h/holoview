#!/usr/bin/env python3
"""Measure source-matched seam displacement; requires numpy and scipy.

No registration is estimated. Match source vertices in their shared metric CRS,
then measure the same pairs in decoded HYPC ECEF. O(N log N) time, O(N) memory
per tile pair. Cached source arrays are derived only from the supplied ZIPs.
"""
import argparse
import itertools
import json
from pathlib import Path
import struct
import zipfile
import numpy as np
from scipy.spatial import cKDTree


def read_hypc(path):
    data = Path(path).read_bytes()
    magic, version, flags, count, upm, *anchor = struct.unpack_from('<4s4I3q', data)
    assert magic == b'HYPC' and version == 2 and upm > 0, path
    start = 44 + (32 if flags & 1 else 0)
    stride = 13 if flags & 2 else 12
    points = np.ndarray((count, 3), dtype='<i4', buffer=data, offset=start,
                        strides=(stride, 4)).astype(np.float64)
    return (points + np.array(anchor, dtype=np.float64)) / upm, upm


def source_points(path, cache):
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / (path.stem + '.npy')
    if out.exists() and out.stat().st_mtime_ns >= path.stat().st_mtime_ns:
        return np.load(out, mmap_mode='r')
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith('.obj')]
        assert len(names) == 1, path
        with archive.open(names[0]) as obj:
            xyz = np.array([list(map(float, line.split()[1:4])) for line in obj
                            if line.split(maxsplit=1)[:1] == [b'v']], dtype=np.float64)
    assert np.isfinite(xyz).all(), path
    np.save(out, xyz)
    return xyz


def stats(d):
    return {'count': len(d), 'median_m': float(np.median(d)),
            'p95_m': float(np.percentile(d, 95)), 'max_m': float(np.max(d))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--before', type=Path, required=True)
    ap.add_argument('--after', type=Path, required=True)
    ap.add_argument('--cache', type=Path, default=Path('target/alignment/source'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--match-m', type=float, default=0.002)
    args = ap.parse_args()
    tiles = {}
    for fixed in sorted(args.after.glob('*.hypc')):
        name = fixed.stem
        src = source_points(args.source / (name + '.zip'), args.cache)
        before, _ = read_hypc(args.before / fixed.name)
        after, upm = read_hypc(fixed)
        assert len(src) == len(before) == len(after), name
        tiles[name] = (src, before, after, upm, src.min(axis=0), src.max(axis=0))
        print(f'{name}: {len(src)} vertices', flush=True)
    pairs = []
    for (an, (a, ab, af, au, amin, amax)), (bn, (b, bb, bf, bu, bmin, bmax)) in itertools.combinations(tiles.items(), 2):
        if np.any(np.maximum(amin, bmin) >
                  np.minimum(amax, bmax) + args.match_m):
            continue
        # Full 3D matching avoids equating a roof and road sharing XY.
        dist, bi = cKDTree(b).query(a, distance_upper_bound=args.match_m, workers=-1)
        ai = np.flatnonzero(np.isfinite(dist))
        bi = bi[ai]
        if not len(ai):
            continue
        src_dist = dist[ai]
        old_dist = np.linalg.norm(ab[ai] - bb[bi], axis=1)
        new_dist = np.linalg.norm(af[ai] - bf[bi], axis=1)
        # Allow projection metric scale differences up to 1% plus lattice errors.
        bound = src_dist * 0.01 + np.sqrt(3) * 0.5 * (1/au + 1/bu) + 1e-7
        assert np.all(np.abs(new_dist - src_dist) <= bound), (an, bn, new_dist.max(), bound.max())
        result = {'tiles': [an, bn], 'source': stats(src_dist),
                  'before': stats(old_dist), 'after': stats(new_dist),
                  'distance_error': stats(np.abs(new_dist - src_dist))}
        pairs.append(result)
        print(json.dumps(result), flush=True)
    assert pairs, 'No seam correspondences; evidence is insufficient'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'source_match_tolerance_m': args.match_m,
                                     'tiles': len(tiles), 'pairs': pairs}, indent=2) + '\n')


if __name__ == '__main__':
    main()
