# Token pruning

Token pruning removes low-salience video tokens in selected steps. It targets model-level redundancy in long spatiotemporal sequences.

## In Sol-Engine

LTX-2.3 uses token pruning in the full optimization stack together with KWL fusion, cache, PISA sparse attention, and NVFP4.

## Practical notes

- Prune only in steps where visual sensitivity is acceptable.
- Keep prompt, seed, scheduler, and resolution fixed when comparing variants.
- Inspect videos side by side; scalar metrics are not enough for final acceptance.
