# ASCIIgent: Training-Free Agentic ASCII Art Generation via Active Visual Perception

**Modern Computer Vision — Final Project Report**  
*Hana Hasan, Amit Arad, Roei Levi, Tom Shur*  

---

## Abstract
While Vision-Language Models (VLMs) have demonstrated impressive performance in vector sketch generation, generating discrete symbolic art (ASCII art) remains an open challenge. Standard autoregressive language models suffer from **scanline blindness**: generating top-to-bottom and left-to-right forces models to commit to early row coordinates without spatial foresight for downstream boundaries. In this work, we present **ASCIIgent**, a training-free framework that transforms a frozen multimodal LLM into an autonomous ASCII artist. Rather than emitting monolithic character dumps in one shot, ASCIIgent operates over a stateful character canvas via an expressive **Discrete ASCII Action Space** (Bresenham edge strokes, tonal density ramps, and localized patch manipulators) driven by an **Active Perception Loop** that iteratively renders intermediate bitmaps, critiques spatial proportions, and performs targeted structural refinements. Across seen and held-out semantic categories, our calibrated benchmark confirms that active visual feedback overcomes autoregressive scanline failure while remaining robust to character-text confounding.

---

## 1. Introduction & Motivation

Humans draw pictures by tracing outlines, assessing global proportions, shading tonal regions, and iteratively erasing or adjusting details. In contrast, standard Large Language Models (LLMs) tasked with generating ASCII art are forced to emit a monolithic $H \times W$ matrix in a single raster-scan pass. When an LLM places a character at row 3 column 10 (e.g., an animal's ear), it has committed to coordinate geometry that must align with paws at row 20 column 10 hundreds of tokens later—without ever perceiving its own drawing in progress. This **scanline blindness** causes severe anatomical distortion, limb detachment, and spatial collapse.

Inspired by **SketchAgent** (Vinker et al., CVPR 2025), which demonstrated that frozen multimodal models can synthesize vector sketches stroke-by-stroke through language planning and visual dialogue, we investigate whether agentic spatial reasoning can be unlocked in the discrete medium of ASCII art.

Unlike continuous Bézier curves, ASCII art introduces a unique multimodal tension: characters in ASCII are simultaneously **language tokens** and **spatial micro-textures**. A slash (`/`) acts as a directional $45^\circ$ edge; an `@` symbol provides high tonal luminance density; a period (`.`) provides subtle stippling. Recent findings by Wang et al. (2025) and ASCIIBench (Luo et al., 2025) reveal that VLMs evaluating ASCII art heavily bias toward reading character-level lexical patterns rather than judging visual silhouettes.

### Core Contributions:
1. **A Formal Discrete Action Space:** We formulate an action space on a discrete canvas $\mathcal{C} \in \Sigma^{H \times W}$ supporting directional edge rendering, tonal shading ramps, and localized patch replacement, avoiding monolithic generation.
2. **The Active Visual Perception Loop:** A multi-turn agentic feedback architecture where the model visually inspects rendered intermediate bitmaps, detects spatial misalignments, and performs targeted localized revisions.
3. **Calibrated Multimodal Evaluation:** We benchmark VLM-as-Judge and CLIP image embeddings against structure-destroying scrambled controls (*scramble-cells* and *scramble-ink*), guaranteeing that scores measure genuine shape recognizability rather than text memorization.

---

## 2. Methodology

```
┌────────────────────────────────────────────────────────────────────────┐
│                          ASCIIgent Architecture                        │
│                                                                        │
│   [Goal: "Draw a Cat"]                                                 │
│            │                                                           │
│            ▼                                                           │
│   [Turn 0: Hierarchical Spatial Plan]                                  │
│   (Head: [2:8, 14:26], Body: [8:18, 10:30], Tail: [10:17, 28:40])    │
│            │                                                           │
│            ▼                                                           │
│   ┌───> [Turn t: Discrete Drawing Actions]                             │
│   │     (draw_line, shade_density, patch_edit, erase_rect)             │
│   │        │                                                           │
│   │        ▼                                                           │
│   │     [AsciiCanvas State Engine (24x48)]                             │
│   │        │                                                           │
│   │        ▼                                                           │
│   │     [Monospace Exact Subpixel Renderer R(C)]                       │
│   │        │                                                           │
│   │        ▼                                                           │
│   │     [Intermediate Bitmap PNG]                                      │
│   │        │                                                           │
│   └───  [Multimodal VLM Perception & Visual Critique]                  │
│         (Detects misalignments, verifies proportions, issues edits)    │
│            │                                                           │
│            ▼ (finish action emitted)                                   │
│   [Final Coherent ASCII Drawing]                                       │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Canvas Representation
Every drawing is represented as an immutable canvas state $C \in \Sigma^{H \times W}$ of printable ASCII characters ($\Sigma \subset [0\text{x}20, 0\text{x}7E]$). The canvas is rendered to an exact continuous RGB image via a deterministic cell-by-cell monospace renderer $\mathcal{R}(C) \to \mathbb{R}^{H' \times W' \times 3}$.

### 2.2 Discrete ASCII Action Space
Instead of generating the canvas as a raw string, an agent policy $\pi$ manipulates a stateful `AsciiCanvas` via parametric tools:
* `draw_line((r1, c1), (r2, c2), char)`: Traces Bresenham lines with automatic directional character selection (`-`, `|`, `/`, `\`).
* `shade_density((r1, c1, r2, c2), level)`: Fills rectangular regions with a calibrated 10-level tonal luminance ramp:
  $$\text{" "} < \text{"."} < \text{":"} < \text{"-"} < \text{"="} < \text{"+"} < \text{"*"} < \text{"\#"} < \text{"\%"} < \text{"@"}$$
* `patch_edit((r, c), patch)`: Stamps localized subgrid patches, allowing the model to repair local errors without disrupting overall canvas alignment.
* `erase_rect((r1, c1, r2, c2))`: Reverts targeted regions to background blanks.

### 2.3 Agentic Active Perception Loop
The generation proceeds across $K$ dynamic turns:
1. **Turn 0 (Hierarchical Planning):** The agent decomposes the target concept into bounding box spatial extents and executes foundational contours.
2. **Turns $1 \dots K$ (Visual Inspection & Revision):** At each turn $t$, the intermediate canvas is rendered to a PNG bitmap $\mathcal{R}(C_t)$ and appended to the multimodal conversation. Conditioned on its own visual output, the model critiques its proportion and symmetry, issuing localized correction tools (`patch_edit` or `erase_rect`) until emitting `finish`.

---

## 3. Experiments & Results

### 3.1 Experimental Protocol
We partition categories into:
- **Seen:** `cat`, `dog`, `bear`, `horse`
- **Held-out:** `octopus`, `giraffe`, `snail`, `penguin`, `lighthouse`, `cactus`, `teapot`, `bicycle`, `mushroom`, `hot air balloon`
- **Exemplar-only:** `fish`, `house`, `sailboat`, `tree` (disjoint from test set to prevent leakage).

### 3.2 Calibrated Vision Evaluation
To prevent the Wang et al. confound (where judges evaluate lexical text rather than visual form), all scores are calibrated using two structure-destroying controls:
* **Scramble-Cells:** Shuffles all canvas cells, preserving character histogram while destroying all spatial topology.
* **Scramble-Ink:** Shuffles non-blank characters exclusively among occupied coordinates, preserving silhouette while scrambling internal texture.

The **Control Gap** $\Delta = S_{\text{real}} - \max(S_{\text{cells}}, S_{\text{ink}})$ serves as our primary dependent variable. A positive gap confirms the metric measures authentic visual shape.

### 3.3 Quantitative Results

| Category | Generation Arm | Parse Clean (%) | Ink Ratio | Coverage | CLIP Real / Scramble (Control Gap) |
|---|---|---|---|---|---|
| **Cat** | Zero-Shot | 100% | 0.0564 | 0.4922 | 0.2412 / 0.2280 (+0.0132) |
| **Cat** | Few-Shot | 100% | 0.1059 | 0.3819 | 0.2589 / 0.2311 (+0.0278) |
| **Cat** | Procedural | 100% | 0.0964 | 0.5278 | 0.2644 / 0.2340 (+0.0304) |
| **Dog** | Few-Shot | 100% | 0.0938 | 0.4167 | 0.2510 / 0.2390 (+0.0120) |
| **Bear** | Zero-Shot | 100% | 0.1892 | 0.6667 | 0.2380 / 0.2325 (+0.0055) |
| **Horse** | Procedural* | 100% | 0.0964 | 0.5903 | **Text Leakage Detected** (`spells_name=True`) |
| **ASCIIgent (Ours)** | **Multi-Turn Agent** | **100%** | **0.1140** | **0.4850** | **0.2785 / 0.2410 (+0.0375)** |

### 3.4 Key Discoveries
1. **The Autoregressive Scanline Failure:** In monolithic one-shot arms, models repeatedly produce detached appendages or collapse into generic humanoid boxes due to lack of intermediate feedback.
2. **Text Leakage Interception:** Under procedural prompting, models frequently cheat by embedding literal text (e.g., spelling `"HORSE"` inside the animal's silhouette). Our automated detector flagged this failure immediately.
3. **Active Perception Gains:** ASCIIgent achieves the highest CLIP visual alignment and the largest Control Gap (+0.0375), proving that step-by-step visual critique allows the model to detect and correct spatial disproportions before finishing.

---

## 4. Discussion & Submission Checklist

By replacing monolithic raster output with discrete tool manipulation and intermediate visual feedback, ASCIIgent bridges the gap between text-based language models and autonomous spatial artists.

### Final Submission Deliverables:
- [x] **Report (2–3 Pages in LaTeX):** `report/main.tex` and `report/asciigent_report.md`
- [x] **Reproducible GitHub Repository:** Clean git repository with zero-cost guardrails, unit tests, and caching.
- [x] **Calibrated Evaluation Metrics:** Automated model-as-judge and CLIP image embeddings with structure-destroying controls.
- [x] **Extended Architecture:** Discrete action space (`AsciiCanvas`) and multi-turn visual feedback loop (`run_agent_drawing_session`).
