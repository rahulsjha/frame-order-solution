# Notes

## What it does

`order.py` (Python, numpy and Pillow only, no network or GPU):

1. **Describe each frame** by a 48-pixel-wide colour thumbnail. Small thumbnails average
   away noise and compression artefacts but still change smoothly from one frame to the next.
2. **Distance** between two frames is the Euclidean distance between their thumbnails,
   computed for all pairs at once with one matrix product.
3. **Order** is the shortest path through all frames (an open travelling-salesman path):
   - greedy edge matching: link the closest pairs first, never more than two links per
     frame and never a loop, then join the pieces end to end, closest ends first;
   - then 2-opt and Or-opt (moving runs of 1 to 3 frames) on 12-nearest-neighbour lists,
     with a dummy frame at distance 0 so the path's ends can move too.
4. **Direction**: pick which end is the start (see below).

Frame loading runs in a process pool. Memory is dominated by the N x N distance matrix
(float32), so about 40 MB for 3,000 frames.

## How I tested it

No real videos were provided beyond `examples/digits`, so I made my own: ffmpeg test
sources (moving patterns, a Mandelbrot zoom, Game of Life, a pan over a large texture,
a zoom, a hue sweep, clips with a hard cut between two scenes), encoded with H.264 (and
some with VP9 and MPEG-4), decoded to PNG and shuffled. Scored with Kendall's tau and the
share of true neighbours that end up next to each other.

| test | frames | neighbours kept together | time |
|---|---|---|---|
| 7 short clips, 320x240 | 150-200 | 100% | 1.5-6 s |
| two scenes, 640x360 | 1,500 | 100% | 12 s |
| pan, 1920x1080 | 300 | 100% | 7 s locally, 21 s in Docker |
| `examples/digits` | 5 | 1, 2, 3, 4, 5 | <1 s |

## Where it still goes wrong

- **Direction is close to a coin flip.** A path reads the same both ways. I tried reading
  the arrow of time from compression: key frames every N frames, with quality drifting
  between them. Sharpness (Laplacian energy) and block-grid artefacts do jump at key
  frames, but whether they jump up or down depends on the encoder and the content. Across
  19 test encodes the best rule I found (small steps in blockiness tend to go down)
  picked the right way about 60% of the time. It is used only as a tie-breaker. A real
  fix probably needs a learned "arrow of time" model (motion, gravity, things spreading
  out), which I didn't add.
- **Order between scenes.** Within a shot the order is reliable. Across hard cuts the
  shots are chained by whichever ends look most alike, which matches the real order only
  by luck. A video with many cuts will have correct runs in the wrong sequence.
- **Repeated or near-static content.** Long static stretches, or a scene that comes back
  later, give many near-equal distances. The path may then jump between the two
  appearances.
- **Synthetic tests only.** All numbers above are on generated clips; real footage with
  grain, flashes or fast motion will score lower.
- **Very long videos.** The all-pairs distance matrix grows with N squared. Past roughly
  20,000 frames it would need a nearest-neighbour index instead.
