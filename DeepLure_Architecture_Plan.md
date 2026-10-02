# 🧵 DeepLure Saree AI Agent — REVISED Architecture & Build Plan v2.0

> **Anti-hallucination edition. Every statement is classified as FACT, DERIVED, HYPOTHESIS, ASSUMPTION, or DECISION.**
> **No training begins until Steps 1–5 (Dataset Audit + Ground Truth) are completed and documented.**

---

## 🔴 Hard Rules (Non-Negotiable)

> [!IMPORTANT]
> These rules override every engineering convenience in this document.

1. **Filename ≠ Design ID.** `img_981759.jpg` is a file identifier only. It proves nothing about design identity, color, relationship to any other file, or source grouping. Never treat it otherwise.
2. **Pseudo-labels are forbidden.** No model training, no metric reporting, and no evaluation may proceed on fabricated or assumed labels. If ground truth does not exist, say so explicitly: `UNKNOWN`.
3. **Every claim is classified.** Each dataset statement must carry one of: `FACT | DERIVED | HYPOTHESIS | ASSUMPTION | DECISION`.
4. **No "same-class = same-design" shortcut.** Banarasi/Bandhani/Ikat/Pichwai are style categories, not individual design identities. A Banarasi floral ≠ a Banarasi geometric. This distinction is central to the task.
5. **Ground truth first.** The sequence is: audit → establish identity → split → build model → evaluate. Not the reverse.
6. **Efficiency numbers are targets until measured.** Nothing in this plan is a "result" until it has been measured and logged with hardware details.

---

## 1. Problem Definition (Unchanged — Correct)

| Field | Value |
|---|---|
| **Objective** | Identify saree designs by surface pattern, independent of color palette |
| **Identification** | Given query image, rank gallery by design similarity |
| **Verification** | Given pair of images, decide SAME_DESIGN or DIFFERENT_DESIGN |
| **Core invariance** | Same motif in different colors → match. Different motifs in same color → no match |
| **Framework** | PyTorch (mandatory) |
| **Compute** | Kaggle Notebooks (free GPU) |

**What "design identity" means for this task:**
- Two images show the SAME DESIGN if they share the same surface motif/weave pattern, regardless of colorway
- Two images are DIFFERENT DESIGNS if their motif structure differs, even if their colors are identical
- Style class (Banarasi/Ikat) is a coarse category, NOT a design identity

---

## 2. Dataset Audit (MANDATORY FIRST STEP)

> [!CAUTION]
> **No model training begins until this section is completed with verified facts.**
> Every cell in this table must be filled with evidence, not assumptions.

### 2.1 Dataset A — Kaggle "Indian Saree Patterns" (`/kaggle`)

| Property | Value | Classification |
|---|---|---|
| Source | Roboflow Universe (`div-szivu/indian-fabric-patterns`) | FACT |
| License | MIT | FACT |
| Total images | 1,468 | FACT (verified by directory count) |
| Pre-processing applied | Resize to 640×640 (stretch), EXIF orientation strip | FACT (from README.roboflow.txt) |
| Augmentation applied | Salt-and-pepper noise, 3× per source image | FACT (from README.roboflow.txt) |
| Split structure | train / valid / test folders | FACT |
| Classes | Banarasi (489), Bandhani (316), Ikat (342), Pichwai (321) | FACT |
| Labels represent | **Textile style/weave type** | FACT |
| Labels represent design identity? | **NO — style class ≠ design ID** | FACT |
| Source image count | ~489 unique sources (1,468 ÷ 3) | DERIVED (arithmetic) |
| Source grouping rule | **UNKNOWN — must be programmatically verified** | ⚠️ UNKNOWN |

**Roboflow Source Grouping — Required Verification Step:**

Before treating the 3× augmentations as "same source" pseudo-identities, we MUST prove it programmatically:

```python
# Required: extract source ID from Roboflow filename pattern
# Example Roboflow filename:
# "0X2A9236_ce21044d-dd2f-4de2-aaf0-21fa02aca371_jpg.rf.0d0755bfccea58800b11bcbea2d26a0f.jpg"
# Pattern: {source_stem}_jpg.rf.{hash}.jpg
# The source_stem before "_jpg.rf." identifies the original image

# Agent MUST:
# 1. Parse all filenames in train/valid/test
# 2. Extract source_stem for each file
# 3. Count files per source_stem
# 4. Verify that groups are 1, 2, or 3 (not arbitrary)
# 5. Output: source_grouping_audit.csv with columns:
#    [source_stem, class, train_count, valid_count, test_count, total]
```

**This audit must produce a table like:**

| source_stem | class | train | valid | test | total | verified |
|---|---|---|---|---|---|---|
| 0X2A9236_ce21044d... | Banarasi | 3 | 0 | 0 | 3 | ✅ |
| 9vvai_512 | Banarasi | 3 | 0 | 0 | 3 | ✅ |
| ... | ... | ... | ... | ... | ... | ... |

**Only after this table exists can we legitimately claim "3 augmentations = same source".**

**⚠️ Data Leakage Warning:** If source groups are split across train/valid/test (e.g. A1→train, A2→val, A3→test), that is leakage. Splits must be at the **source group level**, not the image level.

**Correct leakage-safe split procedure:**
```
Source groups
     ↓
Split source groups 70/15/15
(NOT individual images)
     ↓
Assign all images in each source group to same split
```

---

### 2.2 Dataset B — DeepLure Handloom Corpus (`/handloom_sarees`)

| Property | Value | Classification |
|---|---|---|
| Source | Proprietary 3rd-party vendors (via DeepLure) | FACT |
| Total files | 165 JPEG files | FACT |
| File naming | `img_XXXXXX.jpg` (156 files), `h_img_XXXXXX.jpg` (9 files) | DERIVED (directory scan) |
| Labels provided | **NONE** | FACT |
| Design identity mapping | **UNKNOWN** | FACT |
| h_img prefix meaning | UNKNOWN — could be handloom variant, could be arbitrary | HYPOTHESIS (unverified) |
| img prefix meaning | UNKNOWN | HYPOTHESIS (unverified) |
| Same design pairs | **UNKNOWN — no ground truth exists** | FACT |
| File size range | 4 KB – 573 KB (high variance) | DERIVED |
| Folder structure | Flat — all files in one directory | FACT |
| Intended role (train/eval) | **NOT STATED IN ASSIGNMENT — must be decided after audit** | DECISION REQUIRED |

**What the assignment actually says (verbatim):** *"You may combine, re-split, clean, or relabel the data as needed — document whatever you do."*

This means the handloom corpus is NOT pre-assigned as evaluation-only. Its role is an engineering decision we must justify.

**Required: Handloom Visual Inspection Report**

Before assigning any role to the handloom corpus, a human must:
1. Visually inspect a random sample of 30+ images
2. Answer: Are there any obviously repeated designs? Any clear pairs?
3. Answer: What is the image quality distribution?
4. Answer: Do images look like product photos, editorial shots, or raw fabric scans?
5. Answer: Is there metadata embedded in EXIF that reveals source/design info?

```python
# Required audit script: data/audit/inspect_handloom.py
# Outputs:
#   handloom_audit_report.md  — human-readable summary
#   handloom_exif.csv         — EXIF data for all 165 images
#   handloom_image_stats.csv  — width, height, file size, color histogram mean
```

---

### 2.3 Ground Truth Availability Report (REQUIRED BEFORE TRAINING)

The agent must produce this report. Every row must have evidence:

```
GROUND TRUTH AVAILABILITY REPORT
==================================

Question 1: Do verified same-design pairs exist?
  Kaggle dataset: NO (style class ≠ design ID)
  Handloom corpus: UNKNOWN (no labels, no metadata, no annotation)
  Combined: NO verified design-level pairs currently exist.

Question 2: Can we compute legitimate Recall@K / mAP?
  Current status: NO — requires verified design identities per query
  Path to YES: Manual annotation OR verified source grouping

Question 3: Can we compute legitimate ROC-AUC / EER?
  Current status: NO — requires verified positive/negative pairs
  Path to YES: Manual annotation of minimum viable pair set (see §2.4)

Question 4: Can we legitimately train ArcFace with design identities?
  Current status: NO — no design-level labels exist
  Path to YES: Establish source groupings (Kaggle) + annotation (handloom)

Question 5: What CAN we do now without fabricating labels?
  A. Auxiliary classification (4 style classes, Kaggle only)
  B. ImageNet pretrained embedding extraction — zero-shot baseline
  C. Visual inspection of embedding space (t-SNE)
  D. Self-supervised pretraining (no labels required)
  E. Metric learning on SOURCE GROUPS once §2.1 grouping is verified
```

---

### 2.4 Ground Truth Creation Plan (Replaces "Protocol B aspirational")

> [!IMPORTANT]
> Manual annotation is **primary**, not aspirational. Without it, all metrics are fabricated.

**Minimum viable annotation set:**

| Pair type | Count | How to create | Priority |
|---|---|---|---|
| Verified SAME design pairs (Kaggle, same source group) | ~50 | Programmatic (after source group audit in §2.1) | HIGH |
| Verified DIFFERENT design pairs (Kaggle, different source groups, same class) | ~50 | Programmatic — hard negatives within same style | HIGH |
| Verified DIFFERENT design pairs (Kaggle, different class) | ~100 | Programmatic | MEDIUM |
| Handloom verified pairs (manual) | 20-30 | Human visual inspection | HIGH (required for handloom eval) |

**Ground truth file format (data/ground_truth/):**

```
data/
  ground_truth/
    design_groups.csv   — design_id, image_path, evidence_type
    pairs.csv           — image_a, image_b, label(0/1), source, evidence
    README.md           — explains how each label was established
```

**pairs.csv schema:**
```csv
image_a,image_b,label,source,evidence
kaggle/train/Banarasi/A1.jpg,kaggle/train/Banarasi/A2.jpg,1,roboflow_source_group,same source_stem verified
kaggle/train/Banarasi/A1.jpg,kaggle/train/Banarasi/B1.jpg,0,programmatic,different source_stem same class
handloom/img_001.jpg,handloom/img_002.jpg,1,manual_annotation,same floral motif visible
```

**Evidence types (ONLY these are acceptable):**
- `roboflow_source_group` — same source_stem, verified programmatically
- `dataset_provided` — label came directly from dataset
- `manual_annotation` — human visually confirmed
- `exif_metadata` — device/timestamp metadata proves same shoot
- `programmatic_negative` — different source_stem, verifiably different

**NEVER acceptable:**
- `filename_assumed` — filenames don't prove identity
- `ai_guessed` — model output used as label
- `similarity_threshold` — using the model to create its own training labels

---

## 3. Data Leakage Control Policy

> [!CAUTION]
> Leakage invalidates all evaluation metrics. This policy is mandatory.

### 3.1 Kaggle Leakage

**Problem:** Roboflow applied 3× augmentation per source. All 3 must go into the SAME split.

**Current state of Roboflow splits:** The existing train/valid/test split was done BY ROBOFLOW, not by us. We must verify whether Roboflow split at source level or image level.

```python
# Required check: data/audit/check_leakage.py
# For each source_stem, check which splits it appears in
# Output: leakage_report.csv
# If source_stem appears in multiple splits → leakage detected → must re-split
```

**Correct protocol:**
```
source_stem A → all 3 augmentations → train OR valid OR test (never split across)
```

### 3.2 Train/Gallery/Query Leakage

When constructing the gallery and query sets for evaluation:
- The model must NOT have seen query images during training
- If using Kaggle for training and handloom for evaluation → no leakage (different datasets)
- If using same dataset for both → query images must be from a held-out split, enforced at source group level

---

## 4. Revised Architecture (Corrected)

### 4.1 Correct Framing: Auxiliary Learning → Domain Adaptation → Design Retrieval

The previous plan incorrectly presented ArcFace on 4 style classes as the main metric-learning solution. It is not. It is **auxiliary pretraining** only.

```
STAGE 1: Auxiliary Pretraining (verified labels exist)
    ImageNet pretrained EfficientNet-B3
         ↓
    4-class style classification (Banarasi/Bandhani/Ikat/Pichwai)
    Loss: Standard cross-entropy (NOT ArcFace — class boundaries are too coarse)
    Goal: Adapt backbone from ImageNet → textile domain

STAGE 2: Source-Group Metric Learning (IF source grouping is verified)
    Source groups = ~489 pseudo-identities (ONLY after §2.1 audit confirms grouping)
    Loss: ArcFace OR Triplet Loss on source groups
    Goal: Learn "same original image in different augmentations should be similar"
    ⚠️ Caveat: source groups are augmentation of same image, NOT same design in different colors

STAGE 3: Design-Level Metric Learning (ONLY when design labels exist)
    Requires: ground_truth/design_groups.csv with verified labels
    Loss: ArcFace on design IDs
    Goal: True design-level embedding — the actual task

STAGE 4: Retrieval + Verification
    Gallery: images with verified design IDs
    Query: held-out images with known design IDs
    Metrics: Recall@K, mAP, ROC-AUC, EER (legitimate, not fabricated)
```

### 4.2 Backbone Selection — Initial, Not Final

| Backbone | Params | FLOPs | Why considered |
|---|---|---|---|
| **EfficientNet-B3** (initial) | ~10.7M | ~1.8B | Good accuracy/compute balance, texture sensitivity |
| ResNet-18 (baseline) | ~11.7M | ~1.8B | Simple baseline to beat; fast to train |
| EfficientNet-B0 (lean option) | ~5.3M | ~0.4B | If efficiency is critical |
| ConvNeXt-Tiny | ~28M | ~4.5B | Modern alternative if B3 underperforms |

**Decision:** EfficientNet-B3 is selected as the **initial backbone** because it offers a reasonable accuracy/compute trade-off and has ImageNet-pretrained weights available via torchvision. This is not a claim that it is universally best — performance must be validated against at least one baseline (ResNet-18 minimum).

### 4.3 Embedding Head (Unchanged — Correct)

```
backbone output [B, 1536]
     ↓  Dropout(0.3)
     ↓  Linear(1536, 512, bias=False)
     ↓  BatchNorm1d(512)
     ↓  L2Normalize()
embedding [B, 512] — unit hypersphere
```

### 4.4 Loss Function — Tiered by Label Availability

| Stage | Label quality | Loss function | Why |
|---|---|---|---|
| Auxiliary pretraining | 4 style classes (coarse) | CrossEntropyLoss | Stable, well-understood, matches available labels |
| Source-group metric learning | ~489 augmentation groups (if verified) | Triplet Loss OR ArcFace | Triplet: no class boundary needed. ArcFace: tighter clusters |
| Design-level metric learning | Verified design IDs (after annotation) | ArcFace (m=0.5, s=64) | Angular margin, state-of-art for fine-grained retrieval |

**ArcFace on 4 style classes is NOT the main solution** — it is used only if no better labels exist, and its limitations must be explicitly reported.

### 4.5 Color Invariance — Progressive Experiments, Not All-at-Once

> [!NOTE]
> Start with a minimal baseline. Add complexity only when the baseline is measured.

**Experiment A (Baseline):** Standard RGB + minimal augmentation
```python
transforms.Compose([
    transforms.Resize(256), transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
```

**Experiment B:** Strong color jitter
```python
# Same as A but: saturation=0.8, hue=0.3
```

**Experiment C:** Strong color jitter + RandomGrayscale
```python
# Experiment B + transforms.RandomGrayscale(p=0.3)
```

**Experiment D:** Grayscale-dominant
```python
# Experiment B + transforms.RandomGrayscale(p=0.7)
```

**Ablation table to fill in (not pre-filled):**

| Experiment | Recall@1 | mAP | Notes |
|---|---|---|---|
| A: Baseline RGB | TBD | TBD | Measure this first |
| B: Strong ColorJitter | TBD | TBD | Does more jitter help? |
| C: ColorJitter + Grayscale(0.3) | TBD | TBD | Does grayscale help? |
| D: Grayscale-dominant(0.7) | TBD | TBD | How much is too much? |

**Flips and rotation — must be validated, not assumed:**

```
RandomHorizontalFlip:
  Justification: Fabric photos are sometimes mirrored. Low risk.
  Status: INCLUDED in baseline

RandomVerticalFlip:
  Justification: UNKNOWN — saree borders and pallus may be directional.
  Status: NOT included in baseline. Added only in Experiment E to test.
  
RandomRotation:
  Justification: Slight angle variation in product photography.
  Status: Start with ±5°, test ±15° separately. Cap at 15° max.
```

---

## 5. Evaluation Protocol (Corrected — No Fabricated Metrics)

### 5.1 What Can Be Legitimately Measured

| Metric | Requires | Available now? | Path to availability |
|---|---|---|---|
| Style classification accuracy | 4 class labels | ✅ YES | Use Kaggle test split |
| Recall@K (style-level proxy) | Same style = "same design" | ⚠️ MISLEADING | **Do not present as design retrieval** |
| Recall@K (design-level, true) | Verified design IDs | ❌ NO | Source group audit + annotation |
| mAP (true) | Verified design IDs | ❌ NO | Same |
| ROC-AUC, EER | Verified pos/neg pairs | ❌ NO | Manual annotation (§2.4) |
| t-SNE visual quality | Embeddings only | ✅ YES | Qualitative, not quantitative |
| Zero-shot baseline | Pretrained model, no fine-tuning | ✅ YES | Good starting point |

### 5.2 Style-Level Proxy Evaluation (with explicit caveat)

The following evaluation uses "same style class = positive match" as a proxy. **This is NOT the same as design-level evaluation.** It measures how well the model groups style categories, not individual designs.

```
Query: test split (60 images)
Gallery: train + valid (1,408 images)
Match criterion: SAME STYLE CLASS
                ⚠️ This overestimates real-world performance
                ⚠️ Report this with explicit caveat
```

### 5.3 True Design-Level Evaluation (after annotation, §2.4)

```
Query: images from verified design groups (held-out portion)
Gallery: images from verified design groups (reference portion)
Match criterion: SAME VERIFIED DESIGN ID
                ✅ Legitimate. Report without caveat.
```

### 5.4 Gallery/Query Split — Correct Procedure

**Old (wrong):**
```python
random.seed(42)
query = random.sample(all_images, 0.2)   # random 20%
```

**New (correct):**
```
Step 1: Establish design IDs for all images (from ground_truth/design_groups.csv)
Step 2: For each design ID, require at least 2 images
Step 3: Gallery = first image of each design ID
Step 4: Query = remaining images of each design ID
Step 5: Every query has exactly one known correct gallery match (at minimum)
Step 6: Verify no source-group leakage between gallery and query
```

---

## 6. Experiment Sequence (The Correct Order)

```
STEP 1: Dataset Audit (§2.1 + §2.2)
   → Run data/audit/inspect_handloom.py
   → Run data/audit/parse_roboflow_groups.py
   → Produce: source_grouping_audit.csv, handloom_audit_report.md
   GATE: Agent stops if ground-truth availability is fully unknown

STEP 2: Leakage Check (§3)
   → Run data/audit/check_leakage.py
   → Produce: leakage_report.csv
   → Re-split if leakage detected (source-group-level split)
   GATE: No model training until leakage is confirmed zero

STEP 3: Ground Truth Creation (§2.4)
   → Parse Roboflow source groups (automated)
   → Manual annotation of handloom pairs (20-30 pairs minimum)
   → Produce: data/ground_truth/design_groups.csv, data/ground_truth/pairs.csv

STEP 4: Zero-Shot Baseline
   → Extract embeddings from ImageNet pretrained EfficientNet-B3 (no fine-tuning)
   → Compute proxy Recall@K (style-level, with caveat)
   → Plot t-SNE
   → This is the floor — every model must beat this

STEP 5: Experiment A — Style Classification Baseline
   → Fine-tune EfficientNet-B3 with 4-class CrossEntropy (Kaggle only)
   → Extract embeddings, compute proxy Recall@K
   → Compare to zero-shot (Step 4)
   → Does domain adaptation to textiles help?

STEP 6: Experiment B — Source-Group Metric Learning
   → ONLY if §2.1 confirms valid source groups
   → ArcFace/Triplet on source groups (~489 pseudo-IDs)
   → Compare to Step 5

STEP 7: Color Invariance Ablation (§4.5)
   → Run Experiments A, B, C, D
   → Fill in ablation table
   → Pick best augmentation strategy based on measurements

STEP 8: Design-Level Metric Learning (if ground truth from Step 3 exists)
   → ArcFace on verified design IDs
   → This is the target model

STEP 9: Final Evaluation
   → Recall@K, mAP (design-level, using ground_truth/pairs.csv)
   → ROC-AUC, EER (using verified positive/negative pairs)
   → Report target vs measured for efficiency metrics

STEP 10: Inference + Gallery Search
   → Build gallery index from verified design images
   → Run query inference

STEP 11: Kaggle Notebook (end-to-end, runnable)

STEP 12: Approach note + Efficiency report (with measured numbers)
```

---

## 7. The Ground Truth Manager Module (New — Required)

```
data/
  audit/
    parse_roboflow_groups.py    # Extract source groups from Roboflow filenames
    inspect_handloom.py         # Image stats, EXIF, visual sample
    check_leakage.py            # Verify source groups don't span splits
  ground_truth/
    design_groups.csv           # design_id, image_path, evidence_type
    pairs.csv                   # image_a, image_b, label, source, evidence
    README.md                   # Provenance for every label
  splits/
    source_splits.csv           # source_stem → train/valid/test assignment
    gallery_query.csv           # image_path → gallery/query assignment
```

---

## 8. Revised Repository Structure

```
DeepLure_Saree_AI_Agent/
├── README.md
├── requirements.txt
├── DeepLure_Architecture_Plan.md   ← THIS FILE (v2.0)
├── configs/
│   └── config.yaml
├── data/
│   ├── audit/                      ← NEW: must run before anything else
│   │   ├── parse_roboflow_groups.py
│   │   ├── inspect_handloom.py
│   │   └── check_leakage.py
│   ├── ground_truth/               ← NEW: provenance-tracked labels
│   │   ├── design_groups.csv
│   │   ├── pairs.csv
│   │   └── README.md
│   ├── splits/                     ← NEW: source-group-safe splits
│   │   ├── source_splits.csv
│   │   └── gallery_query.csv
│   ├── dataset.py
│   ├── transforms.py
│   └── splits.py
├── models/
│   ├── backbone.py                 ← Keep
│   ├── embedding_head.py           ← Keep
│   └── arcface.py                  ← Keep (but used at Stage 2/3 only)
├── experiments/                    ← NEW: one script per ablation experiment
│   ├── exp_a_baseline.py
│   ├── exp_b_strong_jitter.py
│   ├── exp_c_grayscale.py
│   ├── exp_d_grayscale_dominant.py
│   └── compare_experiments.py
├── train.py
├── evaluate.py                     ← Corrected (no fabricated metrics)
├── inference.py
├── gallery_builder.py
├── notebooks/
│   └── DeepLure_Kaggle.ipynb
├── utils/
│   ├── metrics.py
│   ├── visualization.py
│   └── logger.py
└── reports/
    ├── dataset_audit_report.md     ← NEW: facts only
    ├── ground_truth_availability.md← NEW: what we know / don't know
    ├── approach_note.md
    └── efficiency_report.md        ← Updated: target vs measured table
```

---

## 9. Efficiency Targets vs. Measured (Corrected)

> [!WARNING]
> The numbers below are **TARGETS**, not results. The measured column must be filled after running the model.

| Metric | Target | Measured | Hardware | Notes |
|---|---|---|---|---|
| Backbone params | ~10.7M | TBD | — | EfficientNet-B3 standard |
| Total params (backbone+head) | ~12.5M | TBD | — | Verify with `model.count_parameters()` |
| FLOPs (224×224) | ~1.8B | TBD | — | EfficientNet-B3 paper value |
| Inference latency | < 20ms | TBD | Kaggle T4 | Measure with cuda.synchronize() |
| Gallery search (N=1000) | < 5ms | TBD | CPU numpy | Measure with perf_counter() |
| Embedding size | 512-d = 2KB | FACT | — | Fixed by architecture |

---

## 10. Honest Assessment of Current State

### What is solid (keep):
- ✅ Metric learning framing (retrieval + verification) — correct
- ✅ EfficientNet-B3 backbone — good starting point
- ✅ 512-d L2-normalized embeddings — correct
- ✅ Cosine similarity for gallery search — correct
- ✅ ColorJitter + RandomGrayscale direction — correct direction
- ✅ Gallery/query separation concept — correct
- ✅ Repository structure, config, logging — good engineering

### What was wrong (corrected in this document):
- ❌ → ✅ Filename prefix as design identity — **REMOVED. Filenames prove nothing.**
- ❌ → ✅ Random 80/20 gallery/query split — **REMOVED. Split after identity established.**
- ❌ → ✅ ArcFace on 4 style classes as main solution — **Demoted to auxiliary pretraining.**
- ❌ → ✅ "489 pseudo-identities" claimed without proof — **Requires programmatic verification.**
- ❌ → ✅ "DeepLure = evaluation set" assumed — **Role TBD after audit.**
- ❌ → ✅ "Manual annotation aspirational" — **Manual annotation is primary. No GT = no metrics.**
- ❌ → ✅ "Same Kaggle class = same design" for evaluation — **Only valid as style-level proxy with explicit caveat.**
- ❌ → ✅ All color invariance techniques applied at once — **Progressive experiments. Measure each.**
- ❌ → ✅ Vertical flip assumed safe — **Excluded from baseline. Tested separately.**
- ❌ → ✅ Efficiency numbers presented as results — **All marked as TARGETS until measured.**
- ❌ → ✅ "EfficientNet-B3 is best" stated as fact — **Stated as initial choice. Baseline required.**

---

## 11. Classification Legend

Every statement in all project documents must carry one of these:

| Label | Meaning | Example |
|---|---|---|
| **FACT** | Directly observed/measured from data or files | "There are 165 files in handloom_sarees/" |
| **DERIVED** | Computed from facts (arithmetic, programmatic) | "156 begin with img_, 9 begin with h_img_" |
| **HYPOTHESIS** | Possible interpretation, not yet verified | "h_img may represent handloom variants" |
| **ASSUMPTION** | Temporary for experiment, explicitly labeled | "ASSUMPTION: source groups contain 3 images each" |
| **DECISION** | Engineering choice based on evidence, documented | "DECISION: Use EfficientNet-B3 as initial backbone" |
| **UNKNOWN** | No evidence available — do not guess | "Design identity of handloom images: UNKNOWN" |
| **TARGET** | Desired outcome, not yet measured | "Latency target: <20ms" |
| **MEASURED** | Actual result with hardware/version details | "Latency measured: 14ms on Kaggle T4 (PyTorch 2.0)" |

---

*Document version: 2.0 | Revised: 2026-10-02*
*Supersedes: DeepLure_Architecture_Plan.md v1.0*
*Key change: All pseudo-label assumptions removed. Dataset audit required before training.*
*Anti-hallucination classification system added throughout.*
