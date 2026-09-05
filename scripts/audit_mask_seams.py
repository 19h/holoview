#!/usr/bin/env python3
"""Compare SMC1 labels at shared geographic tile edges (nearest mask samples)."""
import argparse
import json
from pathlib import Path
import re
import struct
import numpy as np


def mask(path):
    b = path.read_bytes()
    _, _, flags, count, _ = struct.unpack_from('<4s4I', b)
    offset = 44 + (32 if flags & 1 else 0) + count * (13 if flags & 2 else 12)
    assert flags & 4 and flags & 8 and b[offset:offset+4] == b'GEOT'
    extent = np.array(struct.unpack_from('<4i', b, offset+4))*1e-7
    offset += 20
    assert b[offset:offset+4] == b'SMC1'
    w,h,space,encoding,palette = struct.unpack_from('<HHBBH', b, offset+4)
    assert space == 1
    offset += 12 + 2*palette
    length, = struct.unpack_from('<I', b, offset)
    payload = b[offset+4:offset+4+length]
    if encoding == 0:
        raw = np.frombuffer(payload, dtype=np.uint8)
    else:
        records = np.frombuffer(payload, dtype=np.dtype([('count','<u2'),('value','u1')]))
        raw = np.repeat(records['value'], records['count'])
    assert len(raw) == w*h
    return extent, raw.reshape(h,w)


def sample(m, coordinates):
    extent, pixels = m
    lon0, lon1, lat0, lat1 = extent
    x=np.rint(np.clip((coordinates[:,0]-lon0)/(lon1-lon0),0,1)*(pixels.shape[1]-1)).astype(int)
    y=np.rint(np.clip((coordinates[:,1]-lat0)/(lat1-lat0),0,1)*(pixels.shape[0]-1)).astype(int)
    return pixels[y,x]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--before',type=Path,required=True)
    ap.add_argument('--after',type=Path,required=True)
    ap.add_argument('--feature-index',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    features={Path(f['properties']['url']).stem:np.array(f['geometry']['coordinates'][0])
              for f in json.loads(args.feature_index.read_text())['features']}
    tiles={}
    for path in args.after.glob('*.hypc'):
        match=re.fullmatch(r'Tile-(\d+)-(\d+)-1-1',path.stem)
        assert match
        tiles[tuple(map(int,match.groups()))]=(path.stem,mask(args.before/path.name),mask(path))
    rows=[]
    for (x,y),(name,old,new) in sorted(tiles.items()):
        for target,edge in [((x+1,y),(1,2)),((x,y+1),(2,3))]:
            if target not in tiles: continue
            other,other_old,other_new=tiles[target]
            ring=features[name]
            t=np.linspace(0.001,0.999,2048)[:,None]
            positions=ring[edge[0]]*(1-t)+ring[edge[1]]*t
            old_rate=float(np.mean(sample(old,positions)!=sample(other_old,positions)))
            new_rate=float(np.mean(sample(new,positions)!=sample(other_new,positions)))
            rows.append({'tiles':[name,other],'samples':len(t),
                         'before_disagreement':old_rate,'after_disagreement':new_rate})
    assert rows
    args.output.write_text(json.dumps(rows,indent=2)+'\n')
    print('edges',len(rows),'mean disagreement before/after',
          np.mean([r['before_disagreement'] for r in rows]),np.mean([r['after_disagreement'] for r in rows]),
          'max after',max(r['after_disagreement'] for r in rows))


if __name__=='__main__': main()
