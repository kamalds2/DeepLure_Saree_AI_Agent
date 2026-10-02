# 📊 DeepLure Saree AI Agent — Dataset Audit Report

**Data Root:** `kaggle`  
**Total Images:** 1468  
**Unique Images:** 1463  
**Duplicates Detected:** 5  
**Cross-Split Leakage Instances:** 0

## Split Breakdown
| Split | Banarasi | Bandhani | Ikat | Pichwai | Total |
|---|---|---|---|---|---|
| `train` | 432 | 279 | 303 | 279 | **1293** |
| `valid` | 43 | 22 | 26 | 24 | **115** |
| `test` | 14 | 15 | 13 | 18 | **60** |

## Resolution Statistics
- **Width:** Min 0px, Max 0px, Mean 0.0px
- **Height:** Min 0px, Max 0px, Mean 0.0px

## Key Findings & Guardrails
- Total 1468 images audited across 3 splits.
- Class distribution: {'Banarasi': 489, 'Bandhani': 316, 'Ikat': 342, 'Pichwai': 321}.
- Detected 0 cross-split duplicate instances.
- Recommendation: Ensure all cross-split duplicates are removed before training.