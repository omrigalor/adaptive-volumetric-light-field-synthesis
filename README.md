# Adaptive Volumetric Light-Field Synthesis

An inspectable research implementation for shaping a phase-only wavefront so different images come into focus at different depths. It is designed around a concrete question: **can one hologram distribute optical energy across multiple target planes while keeping reconstruction error, crosstalk, speckle, and quantization visible?**

The optimizer uses band-limited angular-spectrum propagation, adaptive multi-plane weighting, phase-only projection, and hardware-like phase quantization. Harder target planes receive more weight in the next iteration instead of allowing an easy plane to dominate the objective.

## What it calculates

- a quantized phase map for a spatial light modulator;
- intensity reconstruction at each requested depth;
- normalized reconstruction error;
- energy efficiency inside each target region;
- speckle contrast and background-to-signal crosstalk;
- adaptive per-plane weights and an optimization trace.

## Why it is distinct

Many compact demonstrations optimize one image at one depth. This prototype makes the multi-depth tradeoff explicit, adapts the loss toward planes that reconstruct poorly, and reports optical failure modes alongside the final image. It is a computational research scaffold, not a medical device or a claim of clinical image quality.

## Run

```bash
python -m pip install -e . pytest
python demo.py
pytest -q
```

## Research basis

- Gerchberg & Saxton (1972), *A practical algorithm for the determination of phase from image and diffraction plane pictures*, Optik 35, 237–246.
- Matsushima & Shimobaba (2009), *Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields*, Optics Express 17, 19662–19673. DOI: 10.1364/OE.17.019662.
- Shi et al. (2021), *Towards real-time photorealistic 3D holography with deep neural networks*, Nature 591, 234–239. DOI: 10.1038/s41586-020-03152-0.

## Validation boundary

The tests cover numerical energy preservation, deterministic optimization, phase quantization, valid adaptive weights, and metric behavior. The model has not been calibrated to a physical optical bench, eye box, display stack, or clinical workflow. Hardware aberration correction, color, occlusion, accommodation, safety, and human-factor validation remain separate work.
