# 📊 DeepLure Saree AI Agent — Visual Dataset Audit Report

**Data Root:** `kaggle`  
**Total Valid Images:** 1468  
**Corrupted Files Detected:** 0

## Split & Category Breakdown
| Split | Banarasi | Bandhani | Ikat | Pichwai | Total |
|---|---|---|---|---|---|
| `train` | 432 | 279 | 303 | 279 | **1293** |
| `valid` | 43 | 22 | 26 | 24 | **115** |
| `test` | 14 | 15 | 13 | 18 | **60** |

## Resolution Statistics
- **Width:** Min 0px, Max 0px, Mean 0.0px
- **Height:** Min 0px, Max 0px, Mean 0.0px

## Core Visual Identity Guardrails
- 🟢 Strict Rule: Identity is derived from visual weave/motifs, never filenames or image paths.
- 🟢 Coarse categories (Banarasi, Bandhani, Ikat, Pichwai) serve as auxiliary domain features only.
- 🟢 Color invariance is enforced by training transformations, ensuring color is not used for design matching.