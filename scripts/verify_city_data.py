#!/usr/bin/env python3
"""Independently verify every packed LOD block and source HYPC receipt.
Usage: python3 scripts/verify_city_data.py CACHE CONVERTED_MANIFEST OUTPUT_JSON
Time O(B + N), space O(maximum block size + N); B is total data bytes.
"""
import hashlib
import json
from pathlib import Path
import sys
import time
import zlib


def verify(cache, manifest, output):
    started = time.monotonic()
    catalog = json.loads((cache / 'catalog.json').read_text())
    receipts = json.loads(manifest.read_text())
    expected = {Path(t['file']).name: t for t in receipts['tiles']}
    source_root = Path(catalog['source_root'])
    sources = [n for n in catalog['nodes'] if 'Source' in n['payload']]
    assert {n['payload']['Source']['source']['path'] for n in sources} == set(expected)
    assert len(sources) == len(expected) == catalog['source_tiles']
    assert sum(n['points'] for n in sources) == catalog['source_points']
    packed = sorted((n for n in catalog['nodes'] if 'Packed' in n['payload']), key=lambda n: n['payload']['Packed']['offset'])
    with (cache / 'points.bin').open('rb') as stream:
        assert stream.read(8) == b'HVLODP01'
        for node in packed:
            payload = node['payload']['Packed']
            assert stream.tell() == payload['offset'], 'Pack gap or overlapping blocks'
            data = stream.read(node['points'] * 16)
            assert len(data) == node['points'] * 16
            assert zlib.crc32(data) == payload['crc32'], f'CRC mismatch at {payload["offset"]}'
        assert stream.tell() == catalog['packed_bytes'] and not stream.read(1)
    for node in sources:
        source = node['payload']['Source']['source']
        receipt = expected[source['path']]
        path = source_root / source['path']
        assert path.stat().st_size == source['bytes'] == receipt['bytes']
        assert node['points'] == receipt['points']
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == receipt['sha256'], path
    result = dict(complete=True, source_tiles=len(sources), source_points=catalog['source_points'],
                  packed_nodes=len(packed), packed_bytes=catalog['packed_bytes'],
                  source_sha256_verified=len(sources), packed_crc32_verified=len(packed),
                  catalog_sha256=hashlib.sha256((cache/'catalog.json').read_bytes()).hexdigest(),
                  elapsed_s=time.monotonic()-started)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    verify(*(Path(arg) for arg in sys.argv[1:]))
