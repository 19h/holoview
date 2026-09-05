#!/usr/bin/env python3
"""Verify identical direct labels for every shared global semantic cell."""
import argparse
import itertools
import json
from pathlib import Path
import struct
import numpy as np
from audit_alignment import read_hypc


def cells(path):
    points,_=read_hypc(path)
    data=path.read_bytes()
    _,_,flags,n,_=struct.unpack_from('<4s4I',data)
    assert flags & 2, 'Direct labels are required'
    offset=44+(32 if flags&1 else 0)
    labels=np.ndarray((n,),dtype='u1',buffer=data,offset=offset+12,strides=(13,))
    x,y,z=points.T
    a=6378137.0; f=1/298.257223563; b=a*(1-f); e=f*(2-f); ep=(a*a-b*b)/(b*b)
    p=np.hypot(x,y);theta=np.arctan2(z*a,p*b)
    lat=np.degrees(np.arctan2(z+ep*b*np.sin(theta)**3,p-e*a*np.cos(theta)**3))
    lon=np.degrees(np.arctan2(y,x))
    xi=np.floor(lon/0.000005).astype(np.int64);yi=np.floor(lat/0.000005).astype(np.int64)
    keys=(xi<<32)|(yi&0xffffffff)
    order=np.argsort(keys)
    unique,starts=np.unique(keys[order],return_index=True)
    lo=np.minimum.reduceat(labels[order],starts);hi=np.maximum.reduceat(labels[order],starts)
    assert np.array_equal(lo,hi), 'One tile assigns different labels within a global cell'
    return unique,lo,(xi.min(),xi.max(),yi.min(),yi.max())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tiles',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    tiles={p.stem:cells(p) for p in sorted(args.tiles.glob('*.hypc'))}
    rows=[]
    for (an,(ak,al,ab)),(bn,(bk,bl,bb)) in itertools.combinations(tiles.items(),2):
        if ab[0]>bb[1] or bb[0]>ab[1] or ab[2]>bb[3] or bb[2]>ab[3]:continue
        shared,ai,bi=np.intersect1d(ak,bk,assume_unique=True,return_indices=True)
        if len(shared):
            disagreements=int(np.sum(al[ai]!=bl[bi]))
            rows.append({'tiles':[an,bn],'shared_cells':len(shared),'disagreements':disagreements})
            assert disagreements==0,(an,bn,disagreements,len(shared))
    assert rows,'No shared cells; verification is insufficient'
    args.output.write_text(json.dumps(rows,indent=2)+'\n')
    print('tiles',len(tiles),'overlapping pairs',len(rows),'shared cells',sum(r['shared_cells'] for r in rows),'disagreements',sum(r['disagreements'] for r in rows))


if __name__=='__main__':main()
