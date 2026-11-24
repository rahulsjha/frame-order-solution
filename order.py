#!/usr/bin/env python3
"""Put the shuffled frames of a video back in order.

1. Describe each frame by a small colour thumbnail.
2. Treat "how different do two frames look" as a distance and find the shortest
   path that visits every frame once (greedy edge matching, then 2-opt and Or-opt).
3. Decide which end of the path is the start (see `direction`).
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

THUMB_W = 48          # thumbnail width used for frame similarity
QUALITY_CROP = 256    # side of the centre crop used to measure compression sharpness
NEIGHBOURS = 12       # candidate neighbours per frame for the path search


# ---------------------------------------------------------------- features

def describe(path):
    """Return (thumbnail vector, blockiness) for one frame."""
    with Image.open(path) as im:
        im = im.convert('RGB')
        w, h = im.size
        thumb_h = max(1, round(THUMB_W * h / w))
        thumb = np.asarray(im.resize((THUMB_W, thumb_h), Image.BOX), dtype=np.float32) / 255.0
        # Blockiness of a full-resolution centre crop: how much stronger the pixel steps are
        # on the codec's 4/8/16-pixel block grid than inside the blocks.
        side = min(QUALITY_CROP, w, h)
        left, top = (w - side) // 2, (h - side) // 2
        crop = np.asarray(im.convert('L').crop((left, top, left + side, top + side)), dtype=np.float32)
    return thumb.ravel(), blockiness(crop)


def blockiness(g):
    if min(g.shape) < 17:
        return 0.0
    dh = np.abs(np.diff(g, axis=1))
    dv = np.abs(np.diff(g, axis=0))
    cols = np.arange(dh.shape[1]) + 1
    rows = np.arange(dv.shape[0]) + 1
    ratios = []
    for b in (4, 8, 16):
        on_h, on_v = cols % b == 0, rows % b == 0
        ratios.append(dh[:, on_h].mean() / (dh[:, ~on_h].mean() + 1e-6))
        ratios.append(dv[on_v, :].mean() / (dv[~on_v, :].mean() + 1e-6))
    return float(np.mean(ratios))


def distances(features):
    """Euclidean distance between every pair of thumbnails."""
    x = features.astype(np.float32)
    sq = (x * x).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    np.maximum(d2, 0, out=d2)
    d = np.sqrt(d2)
    np.fill_diagonal(d, 0.0)
    return d


# ---------------------------------------------------------------- path search

class _Sets:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def greedy_path(d):
    """Link the closest pairs first, never giving a frame more than two links or making
    a loop, then join the pieces end to end, closest ends first."""
    n = len(d)
    k = min(n - 1, NEIGHBOURS)
    masked = d + np.diag(np.full(n, np.inf, dtype=d.dtype))
    nbr = np.argpartition(masked, k - 1, axis=1)[:, :k]
    edges = {(min(i, j), max(i, j)) for i in range(n) for j in nbr[i]}
    edges = sorted(edges, key=lambda e: d[e[0], e[1]])
    deg = [0] * n
    adj = [[] for _ in range(n)]
    sets = _Sets(n)

    def link(a, b):
        deg[a] += 1
        deg[b] += 1
        adj[a].append(b)
        adj[b].append(a)
        sets.union(a, b)

    for a, b in edges:
        if deg[a] < 2 and deg[b] < 2 and sets.find(a) != sets.find(b):
            link(a, b)

    while True:
        ends = [i for i in range(n) if deg[i] < 2]
        groups = {sets.find(i) for i in ends}
        if len(groups) <= 1:
            break
        e = np.array(ends)
        sub = d[np.ix_(e, e)].astype(np.float64)
        roots = np.array([sets.find(i) for i in ends])
        sub[roots[:, None] == roots[None, :]] = np.inf
        i, j = np.unravel_index(np.argmin(sub), sub.shape)
        link(int(e[i]), int(e[j]))

    start = next((i for i in range(n) if deg[i] < 2), 0)
    path, prev = [start], -1
    while len(path) < n:
        cur = path[-1]
        nxt = next(x for x in adj[cur] if x != prev)
        prev = cur
        path.append(nxt)
    return path


def improve(path, d):
    """2-opt and Or-opt on the open path. A dummy frame at distance 0 from every frame
    closes the loop, so the path's two ends can move freely too."""
    n = len(path)
    if n < 4:
        return path
    m = n + 1                      # frame n is the dummy

    def D(a, b):
        return 0.0 if a == n or b == n else float(d[a, b])

    k = min(n - 1, NEIGHBOURS)
    masked = d + np.diag(np.full(n, np.inf, dtype=d.dtype))
    near = np.argpartition(masked, k - 1, axis=1)[:, :k]
    nbr = []
    for i in range(n):
        row = sorted(near[i].tolist(), key=lambda j: d[i, j])
        nbr.append([n] + row)      # the dummy is at distance 0 from everyone
    nbr.append(list(range(min(n, k))))
    tour = list(path) + [n]

    def positions():
        p = [0] * m
        for i, c in enumerate(tour):
            p[c] = i
        return p

    for _ in range(50):
        changed = False
        # 2-opt: replace edges (a,b) and (c,e) with (a,c) and (b,e)
        pos = positions()
        for a in range(m):
            i = pos[a]
            b = tour[(i + 1) % m]
            dab = D(a, b)
            for c in nbr[a]:
                if D(a, c) >= dab:
                    break
                j = pos[c]
                e = tour[(j + 1) % m]
                if c == b or e == a:
                    continue
                gain = dab + D(c, e) - D(a, c) - D(b, e)
                if gain > 1e-12:
                    lo, hi = (i + 1, j) if i < j else (j + 1, i)
                    tour[lo:hi + 1] = tour[lo:hi + 1][::-1]
                    for x in range(lo, hi + 1):
                        pos[tour[x]] = x
                    changed = True
                    break
        # Or-opt: move a run of 1-3 frames next to a closer frame
        for seg in (1, 2, 3):
            pos = positions()
            i = 0
            while i < m:
                if seg >= m - 2:
                    break
                s = [tour[(i + t) % m] for t in range(seg)]
                p, q = tour[(i - 1) % m], tour[(i + seg) % m]
                remove_gain = D(p, s[0]) + D(s[-1], q) - D(p, q)
                best = None
                for end, other in ((s[0], s[-1]), (s[-1], s[0])):
                    for c in nbr[end]:
                        if c in s:
                            continue
                        j = pos[c]
                        for c2 in (tour[(j - 1) % m], tour[(j + 1) % m]):
                            if c2 in s:
                                continue
                            add = D(c, end) + D(c2, other) - D(c, c2)
                            if remove_gain - add > 1e-12 and (best is None or remove_gain - add > best[0]):
                                best = (remove_gain - add, c, c2, end)
                if best:
                    _, c, c2, end = best
                    rest = [x for x in tour if x not in s]
                    piece = s if end == s[-1] else s[::-1]   # piece[-1] sits next to c
                    jc = rest.index(c)
                    if rest[(jc + 1) % len(rest)] == c2:
                        rest[jc + 1:jc + 1] = piece
                    else:
                        rest[jc:jc] = piece[::-1]
                    tour[:] = rest
                    pos = positions()
                    changed = True
                i += 1
        if not changed:
            break
    k0 = tour.index(n)
    return tour[k0 + 1:] + tour[:k0]


# ---------------------------------------------------------------- direction

def direction(block):
    """+1 if the path already runs forward in time, -1 if it should be reversed.

    This is the weak part (see NOTES.md). Ordering by similarity gives a path, but a
    path reads the same both ways. The only clue used here is compression: between key
    frames, blockiness tends to fall a little from frame to frame, so we count whether
    the small frame-to-frame steps lean downwards. On synthetic tests it picked the
    right way about 60% of the time, so treat it as a tie-breaker, not a detector."""
    s = np.asarray(block, dtype=np.float64)
    if len(s) < 8 or s.std() == 0:
        return 1
    steps = np.diff(s)
    z = steps / (np.median(np.abs(steps)) + 1e-12)
    small = np.abs(z) < 2          # ignore cuts and key-frame jumps
    score = -np.tanh(z[small]).sum()
    return 1 if score >= 0 else -1


# ---------------------------------------------------------------- main

def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else '/frames'
    names = sorted(f for f in os.listdir(folder) if f.lower().endswith('.png'))
    if len(names) <= 1:
        print('\n'.join(names))
        return
    paths = [os.path.join(folder, f) for f in names]
    with Pool(max(1, os.cpu_count() or 1)) as pool:
        described = pool.map(describe, paths, chunksize=8)
    feats = np.stack([f for f, _ in described])
    block = [q for _, q in described]
    d = distances(feats)
    path = improve(greedy_path(d), d)
    if direction([block[i] for i in path]) < 0:
        path.reverse()
    sys.stdout.write('\n'.join(names[i] for i in path) + '\n')


if __name__ == '__main__':
    main()
