# ML basics for this project

This file explains, from zero, every idea the rest of the project depends on. It assumes you know Python
and have seen a neural network diagram once. Nothing more.

Each concept follows the same pattern:

1. **What it is**, in plain words.
2. **Why we need it here**, tied to cardiac MRI.
3. **A picture or analogy**, usually a Mermaid diagram or a small worked example with real numbers.
4. **Where it lives in our code**, with the file and function name.

Sections are filled in during the phase that first uses the idea, so the explanations always match code
that exists. The outline below shows what is coming and when. If a section has only its one-line summary,
the code for it has not been written yet.

## Outline

| # | Section | What it will explain | Written in |
|---|---|---|---|
| 1 | What an image is to a computer | A grid of numbers; greyscale vs colour; `dtype` and value ranges | Phase 1 |
| 2 | MRI volumes and voxel spacing | A 3D stack of slices; why a voxel is a box in millimetres, not a dot; why thick slices matter | Phase 1 |
| 3 | Cine MRI, end-diastole and end-systole | A 4D scan is a movie of one heartbeat; the two frames we label and why | Phase 1 |
| 4 | Segmentation vs classification | One label per image vs one label per pixel; our four labels (background, RV, myocardium, LV) | Phase 2 |
| 5 | Convolution | A 3×3 filter sliding over a 5×5 image, worked by hand | Phase 2 |
| 6 | Pooling and upsampling | Shrinking to see more context, growing back to draw outlines | Phase 2 |
| 7 | The U-Net | Encoder, decoder and skip connections, drawn with the channel sizes we actually use | Phase 2 |
| 8 | Loss functions | Dice derived by hand on a 4×4 example; cross-entropy as "how surprised was the model" | Phase 2 |
| 9 | The training loop | Forward → loss → backward → optimiser step, and what an epoch is | Phase 2 |
| 10 | Overfitting and validation | Why we hold patients back, and why we split by patient instead of by slice | Phase 2 |
| 11 | Data augmentation | Teaching the model that a slightly rotated heart is still a heart | Phase 2 |
| 12 | Transfer learning | Reusing an encoder trained on everyday photos for grey MRI slices | Phase 3 |
| 13 | Foundation models and prompting | What MedSAM is, what a box prompt is, and why it needs no training | Phase 3 |
| 14 | From masks to millimetres | Counting voxels, multiplying by spacing, and getting millilitres | Phase 4 |
| 15 | Decision trees and gradient boosting | One tree as a flowchart of questions; many small trees correcting each other | Phase 5 |
| 16 | SHAP values | How much each measurement pushed one patient's prediction, and in which direction | Phase 5 |
| 17 | Uncertainty | Why a model can be confidently wrong; test-time augmentation and entropy as a "doubt map" | Phase 5 |

## How to read the rest of the docs

- `architecture.md` describes what the pieces are and how they connect.
- `flow.md` follows real execution step by step, with file and function names.
- `decisions.md` records every design choice: the options, what I picked, and what I gave up.

Start here, then read `flow.md` alongside the code.
