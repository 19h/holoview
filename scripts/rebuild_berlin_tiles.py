#!/usr/bin/env python3
"""Regenerate supplied Berlin OBJ tiles with one EPSG:25833 transform.

Stages vertex references and uniform surface samples, rebuilds geographic masks,
checks independent cs2cs
control samples, and optionally installs with byte-exact backups. Does not infer
registration or the source vertical datum. Requires numpy/scipy, PROJ cs2cs,
and a built obj2hypc converter. All source heights are preserved.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import numpy as np
from audit_alignment import read_hypc, source_points


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def retain_geographic_mask(original, rebuilt, resampled=False):
    old, new = original.read_bytes(), rebuilt.read_bytes()
    oh = struct.unpack_from('<4s4I3q', old)
    nh = struct.unpack_from('<4s4I3q', new)
    assert oh[0:2] == nh[0:2] == (b'HYPC', 2)
    if not resampled:
        assert oh[3] == nh[3], 'Source vertex count changed'
    assert not oh[2] & 2, 'Per-vertex labels require separate source-order validation'
    assert not nh[2] & 2 and oh[2] & 1 and nh[2] & 1
    assert old[44:76] == new[44:76], 'Tile identity changed'
    if not oh[2] & 8:
        return
    assert oh[2] & 4 and nh[2] & 4, 'Geographic mask requires matching GEOT'
    oo, no = 76 + oh[3] * 12, 76 + nh[3] * 12
    assert old[oo:oo+20] == new[no:no+20], 'Semantic geographic extent changed'
    mask = old[oo+20:]
    assert mask[:4] == b'SMC1' and mask[8] == 1, 'Only geographic SMC1 can be retained'
    # Regenerated files have no SMC1 unless this stage has already run.
    if nh[2] & 8:
        assert new[no+20:] == mask
        return
    assert len(new) == no + 20
    header = bytearray(new[:12])
    struct.pack_into('<I', header, 8, nh[2] | 8)
    rebuilt.write_bytes(bytes(header) + new[12:] + mask)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, action='append', required=True)
    ap.add_argument('--sources', type=Path, action='append', required=True)
    ap.add_argument('--feature-index', type=Path, required=True)
    ap.add_argument('--stage', type=Path, default=Path('target/alignment/rebuild'))
    ap.add_argument('--converter', type=Path, default=Path('crates/obj2hypc/target/release/obj2hypc'))
    ap.add_argument('--osm-pbf', type=Path, required=True)
    ap.add_argument('--surface-spacing-m', type=float, default=0.5)
    ap.add_argument('--install', action='store_true')
    args = ap.parse_args()
    args.stage.mkdir(parents=True, exist_ok=True)
    manifest = []
    semantic_source_sha256 = digest(args.osm_pbf)
    for dataset in args.dataset:
        originals = sorted(dataset.glob('*.hypc'))
        assert originals, dataset
        stage = args.stage / dataset.name
        source_dir = stage / 'sources'
        fixed_dir = stage / 'fixed'
        backup_dir = stage / 'original'
        surface_dir = stage / 'surface'
        source_dir.mkdir(parents=True, exist_ok=True)
        fixed_dir.mkdir(parents=True, exist_ok=True)
        backup_dir.mkdir(parents=True, exist_ok=True)
        upms = {}
        for original in originals:
            backup = backup_dir / original.name
            # On repeated install, the backup is authoritative only when the
            # current file is either that original or the previously staged fix.
            if backup.exists():
                assert digest(original) == digest(backup) or ((fixed_dir / original.name).exists()
                    and digest(original) == digest(fixed_dir / original.name)) or (
                    (surface_dir / original.name).exists() and digest(original) == digest(surface_dir / original.name))
            else:
                shutil.copy2(original, backup)
            upm = struct.unpack_from('<I', backup.read_bytes(), 16)[0]
            upms.setdefault(upm, []).append(original.stem)
            source = next((root / (original.stem + '.zip') for root in args.sources
                           if (root / (original.stem + '.zip')).exists()), None)
            assert source is not None, original
            link = source_dir / source.name
            if link.is_symlink():
                assert link.resolve() == source.resolve()
            else:
                link.symlink_to(source.resolve())
        env = dict(os.environ, RAYON_NUM_THREADS='4', RUST_LOG='info')
        for upm, stems in upms.items():
            group = stage / f'sources-{upm}'
            group.mkdir(parents=True, exist_ok=True)
            for stem in stems:
                link = group / (stem + '.zip')
                if not link.exists():
                    link.symlink_to((source_dir / link.name).resolve())
            subprocess.run([str(args.converter.resolve()), '--input-dir', str(group),
                            '--output-dir', str(fixed_dir), '--input-cs', 'projected',
                            '--source-crs', 'EPSG:25833', '--sampling', 'vertices', '--units-per-meter', str(upm),
                            '--feature-index', str(args.feature_index), '--overwrite'], check=True, env=env)
        for original in originals:
            fixed = fixed_dir / original.name
            backup = backup_dir / original.name
            retain_geographic_mask(backup, fixed)
            source = source_dir / (original.stem + '.zip')
            xyz = source_points(source, stage / 'cache')
            points, upm = read_hypc(fixed)
            assert len(xyz) == len(points)
            indices = np.unique(np.linspace(0, len(xyz)-1, min(len(xyz), 4096), dtype=int))
            samples = xyz[indices]
            data = '\n'.join(' '.join(format(v, '.17g') for v in p) for p in samples) + '\n'
            control = subprocess.run(['cs2cs', 'EPSG:25833', 'EPSG:4978', '-f', '%.10f'],
                                     input=data, text=True, capture_output=True, check=True)
            expected = np.array([list(map(float, line.split())) for line in control.stdout.splitlines()])
            assert expected.shape == samples.shape and np.isfinite(expected).all()
            error = np.abs(points[indices] - expected)
            assert error.max() <= 0.5 / upm + 1e-8, (original, error.max())
            row = {'dataset': str(dataset), 'tile': original.name, 'points': len(points),
                   'samples': len(indices), 'max_axis_error_m': float(error.max()),
                   'source_sha256': digest(source), 'original_sha256': digest(backup),
                   'fixed_sha256': digest(fixed), 'source_crs': 'EPSG:25833',
                   'height_offset_m': 0.0, 'semantic_mask': 'retained in original geographic frame'}
            manifest.append(row)
            print(json.dumps(row), flush=True)
        # Resample surfaces and rebuild masks from complete OSM ways. The vertex
        # files above remain as independently checked source-mesh references.
        surface_dir.mkdir(parents=True, exist_ok=True)
        for upm in upms:
            subprocess.run([str(args.converter.resolve()), '--input-dir', str(stage / f'sources-{upm}'),
                            '--output-dir', str(surface_dir), '--input-cs', 'projected',
                            '--source-crs', 'EPSG:25833', '--sampling', 'surface',
                            '--surface-spacing-m', str(args.surface_spacing_m), '--units-per-meter', str(upm),
                            '--feature-index', str(args.feature_index), '--osm-pbf', str(args.osm_pbf),
                            '--overwrite'], check=True, env=env)
        audit = stage / 'surface-audit.json'
        subprocess.run([sys.executable, str(Path(__file__).with_name('audit_surface.py')),
                        '--source', str(source_dir), '--reference', str(fixed_dir),
                        '--surface', str(surface_dir), '--cache', str(stage/'cache'),
                        '--output', str(audit)], check=True)
        verified = {row['tile']: row for row in json.loads(audit.read_text())}
        assert set(verified) == {path.name for path in originals}
        for row in manifest:
            if row['dataset'] == str(dataset):
                result = verified[row['tile']]
                row['surface_points'] = result['surface_points']
                row['surface_samples_checked'] = result['samples']
                row['max_surface_distance_m'] = result['max_surface_distance_m']
                row['installed_sha256'] = result['sha256']
                row['semantic_mask'] = 'rebuilt from complete intersecting OSM ways without vertex clamping'
                row['semantic_source_sha256'] = semantic_source_sha256
                row['semantic_labels'] = 'direct labels on shared global cells'
        subprocess.run([sys.executable, str(Path(__file__).with_name('audit_label_cells.py')),
                        '--tiles', str(surface_dir), '--output', str(stage/'label-cells-audit.json')], check=True)
        # Install only after the whole dataset passed geometry and shared-label controls.

        if args.install:
            for original in originals:
                fixed = surface_dir / original.name
                temp = original.with_suffix('.hypc.rebuild-tmp')
                shutil.copy2(fixed, temp)
                temp.replace(original)
                shutil.copy2(fixed.with_suffix('.provenance.json'), original.with_suffix('.provenance.json'))
    (args.stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
