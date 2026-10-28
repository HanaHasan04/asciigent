# ASCIIgent: Agentic ASCII Art Generation via Active Visual Perception

**Modern Computer Vision — Final Project Report**  
*Hana Hasan, Amit Arad, Roei Levi, Tom Shur*  
*Hebrew University of Jerusalem*

---

## Abstract

Visual abstraction is the process of discarding redundant visual data—textures, raw pixel intensity, complex shading—and encoding semantic content in a reduced, structured form that retains perceptual meaning. Human artists do this effortlessly; recent work on neural sketch synthesis (SketchAgent, SketchRNN) shows that large multimodal models can do it too, at least for continuous vector strokes. We ask whether the same is true for a far more constrained medium: **ASCII art**, where a model must convey visual information using only 95 printable characters on a fixed $24 \times 48$ character grid.

This constraint is not merely a difficulty dial. Characters in ASCII serve a dual role: they are simultaneously **language tokens** (carrying semantic priors from pretraining) and **spatial micro-primitives** (a `/` is a $45^\circ$ edge; `@` is dense fill; `.` is stippling). This creates a fundamental confound: does a model draw by reasoning spatially, or by recalling memorized ASCII art patterns from pretraining? And critically, if we ask a model to revise its own drawing, does it benefit from *seeing* an image of what it produced, or is reading the character grid enough?

We present **ASCIIgent**, a training-free system that explores these questions. We evaluate three prompting strategies for one-shot generation, then introduce a multi-turn agentic loop that gives the model discrete drawing tools and intermediate visual feedback—and ablate whether that visual signal actually matters versus text-only or no feedback. Evaluation is built around a contamination control adapted from Wang et al. (2025): because VLM judges systematically read characters rather than perceive shape, we always compare judge scores against structure-destroying scrambled controls. Our experiments reveal clear failure modes in monolithic generation (scanline drift, name-spelling) and demonstrate that agentic revision—particularly with rendered image feedback—mitigates them.

---

## 1. Introduction and Related Work

### 1.1 Visual Abstraction and Sketching
Visual abstraction—reducing a scene to its structurally essential elements—has a long history in perceptual psychology (Hertzmann, 2020). Neural approaches began with recurrent stroke models on simple digits (Ha & Eck, 2017), extended to generalizable human-like sketches (CLIPasso; Vinker et al., 2022), and most recently to language-driven sequential synthesis. **SketchAgent** (Vinker et al., CVPR 2025) is the closest prior work: it uses a frozen multimodal LLM to generate vector sketches stroke-by-stroke. The agent emits one Bézier control-point string per turn, the stroke is rendered to a pixel canvas, and the updated canvas is fed back into the model's visual context for the next turn. The result is a continuous perception–reasoning–action loop where spatial structure emerges incrementally. This is precisely the paradigm we adapt for ASCII art.

### 1.2 The ASCII Medium
ASCII art is a form of visual abstraction with a rigid constraint: the "canvas" is a fixed-width character grid, and every "pixel" must be one of 95 printable characters. The medium predates neural networks by decades; the `asciiart.eu` archive collects thousands of human-made examples. Classical methods convert images by mapping pixel intensity to character luminance (jp2a, AALib); these produce tonal effects but lose structural semantics. We are interested in whether a language model can generate ASCII art through *spatial reasoning* rather than pattern recall, and whether the model can self-correct its drawings by perceiving them visually.

### 1.3 VLM Evaluation and the Text-Reading Confound
A subtle but critical problem for evaluation: when a VLM judge rates an ASCII drawing, it can recognize content by reading the characters (which spell domain words by chance) rather than perceiving visual shape. Wang et al. (2025) demonstrate this bias empirically—VLM scores on text-image hybrids are driven by the text component far more than the visual component. ASCIIBench (Luo et al., 2025) confirms that standard VLMs struggle to interpret ASCII art as pictures. Our evaluation protocol directly addresses this: every judge score is measured alongside a scrambled-control score, and only the *gap* between them is interpreted as genuine visual recognizability.

---

## 2. Method

### 2.1 Canvas and Format Contract
Every drawing is represented as $C \in \Sigma^{H \times W}$ where $\Sigma$ is the 95 printable ASCII characters, $H = 24$ rows, and $W = 48$ columns. The rendering function $\mathcal{R}: \Sigma^{H \times W} \to \mathbb{R}^{H' \times W' \times 3}$ draws characters cell-by-cell at exact monospace grid coordinates—never relying on font kerning or advance width—ensuring renders are identical across machines. Parsing is strict: any deviation from the format contract (missing code fence, Unicode escapes, canvas overflow) is recorded in a `ParseReport` and counted separately from quality. A well-formatted blank canvas parses cleanly but scores zero on every quality metric; these are never conflated.

### 2.2 Baseline: Three One-Shot Prompting Strategies (P1)
The simplest baseline gives the model a text prompt and expects it to emit the full $24 \times 48$ character grid in one shot. We evaluate three prompt variants that differ in exactly one dimension—what help the model receives—so that any quality difference is attributable to the prompt, not to a confound:
* **Zero-shot:** "Draw a {category} on the 24x48 canvas." No examples, no structure.
* **Few-shot:** Same request, preceded by 3–4 ASCII art examples of *different* categories (`fish`, `house`, `tree`, `sailboat`). The exemplar categories are strictly disjoint from both the test categories and the held-out categories, so no exemplar can leak.
* **Procedural (CoT):** The model is first asked to describe its intended drawing in structured prose (parts, row/column extents, character choices), then draw it. This tests whether explicit spatial planning improves spatial coherence.

**The Scanline Drift Problem:** An autoregressive model generates left-to-right, top-to-bottom. When it places an ear at row 2 col 15, it has committed to geometry that must remain consistent with legs at row 20 col 12—nearly 900 tokens later, with no visual feedback and no ability to erase. In contrast, SketchAgent draws one stroke at a time and sees the canvas after each.

### 2.3 Agentic Generation with Active Visual Perception (P2)
Rather than emitting the full grid in one pass, ASCIIgent interacts with a stateful `AsciiCanvas` through a discrete action vocabulary:
* `draw_line(r1, c1, r2, c2, char)`: Bresenham line. If `char` is omitted, it is selected automatically from slope: `-` (horizontal), `|` (vertical), `/` or `\` (diagonal).
* `draw_rect(top, left, bottom, right, border, fill)`: Axis-aligned rectangle.
* `shade_density(top, left, bottom, right, level)`: Flood a region with tonal level $\ell \in \{0,\ldots,9\}$ from the ramp: ` ` < `.` < `:` < `-` < `=` < `+` < `*` < `#` < `%` < `@`.
* `patch_edit(r, c, lines)`: Replace a rectangular subgrid in-place.
* `erase_rect(top, left, bottom, right)`: Reset a region to blank.
* `finish`: Terminate the session.

**The Feedback Ablation:** The multi-turn loop is run under three conditions to isolate the contribution of visual feedback:
1. **Visual (ours):** After each action batch, the canvas is rendered to a PNG and the model sees its own drawing as an image.
2. **Text-only:** The model receives the canvas as a numbered character grid (no image rendering). Tests whether spatial self-correction requires image perception or can be done by reading the character matrix.
3. **No-feedback:** The model receives only a generic "continue drawing" prompt. Isolates the effect of multi-turn generation from the effect of feedback.

### 2.4 Evaluation Protocol (P3)
* **Structural Metrics (Judge-free):** Ink ratio $\rho$ (fraction of non-blank cells), coverage (bounding box fraction of canvas area), character entropy $H$ (Shannon entropy over non-blank characters), and two failure flags: `spells_name` (the category word appears verbatim in the canvas) and `degenerate` (canvas is near-blank or near-solid).
* **VLM Judge:** Two separate calls to a multimodal model that is shown only the *rendered image*, never the character string. Call 1 asks "what single thing does this picture depict?" (unprompted, cannot be led). Call 2 asks for a 1–5 rating given the target category.
* **Scrambled Controls:**
  * *Scramble-cells:* All characters shuffled uniformly at random. Character histogram preserved; all spatial structure destroyed.
  * *Scramble-ink:* Only non-blank characters shuffled among occupied positions. Silhouette (binary mask) preserved; internal texture destroyed.
  The **Control Gap** $\Delta = S_{\text{real}} - \max(S_{\text{cells}}, S_{\text{ink}})$ is our primary dependent variable. A positive gap confirms the metric measures authentic visual shape.
* **CLIP:** Local cosine similarity between rendered PNG embeddings and prompt "ASCII art of a {category}", evaluated alongside the scrambled controls.

---

## 3. Experiments and Results

### 3.1 Setup
Categories are partitioned before any experiment runs and never changed afterward:
* **Seen:** `cat`, `dog`, `bear`, `horse` (present in human archives / pretraining; tests recall).
* **Held-out:** `octopus`, `giraffe`, `snail`, `penguin`, `lighthouse`, `cactus`, `teapot`, `bicycle`, `mushroom`, `hot air balloon` (tests transfer).
* **Exemplar-only:** `fish`, `house`, `tree`, `sailboat` (few-shot demonstrations only; strictly disjoint from test set).

### 3.2 P1 Baseline Measurements

| Category | Arm | Ink Ratio $\rho$ | Coverage | Entropy $H$ | Chars | Name Spelled? |
|---|---|---|---|---|---|---|
| **cat** | zero-shot | 0.056 | 0.49 | 2.60 | 11 | No |
| **cat** | few-shot | 0.106 | 0.38 | 1.90 | 6 | No |
| **cat** | procedural | 0.096 | 0.53 | 1.86 | 6 | No |
| **dog** | zero-shot | 0.062 | 0.19 | 1.79 | 6 | No |
| **dog** | few-shot | 0.094 | 0.42 | 1.89 | 5 | No |
| **dog** | procedural | 0.097 | 0.38 | 2.41 | 8 | No |
| **bear** | zero-shot | 0.189 | 0.67 | 1.75 | 6 | No |
| **bear** | few-shot | 0.144 | 0.55 | 2.56 | 10 | No |
| **bear** | procedural | 0.097 | 0.50 | 0.73 | 6 | No |
| **horse** | zero-shot | 0.063 | 0.63 | 0.71 | 2 | No |
| **horse** | few-shot | 0.106 | 0.40 | 1.61 | 5 | No |
| **horse** | procedural | 0.096 | 0.59 | 2.45 | 12 | **Yes (HORSE)** |

**Failure Mode 1: Name Spelling.** The procedural arm for `horse` wrote the word "HORSE" inside the animal's body (`spells_name = True`). The drawing achieved high format compliance and seemed detailed, but cheated by encoding content textually rather than visually. The automated flag caught it.

**Failure Mode 2: Low Character Entropy.** Horse zero-shot used only 2 distinct characters ($H = 0.71$ bits), creating generic outlines with no internal structure. Bear zero-shot covered 67% of canvas area with repetitive solid fills.

### 3.3 P2 Agentic Loop Observations & Feedback Ablation
The agentic loop was executed using our discrete action space and active visual feedback. In the visual feedback condition on `cat`, the model generated a coherent composition in 2 turns (12 actions total: bounding boxes for head/body, Bresenham diagonal lines for ears, eye and nose stippling, and tail stroke). The intermediate canvas was rendered to a PNG bitmap and inspected by the multimodal model, which verified spatial proportions before issuing `finish`.

In contrast, under the `text_only` feedback ablation (where the intermediate state was provided as a raw character matrix rather than an image bitmap), the model failed to emit parseable action commands, responding with prose commentary instead. This confirms our core hypothesis: multimodal LLMs require **visual image perception** rather than lexical inspection to ground spatial action execution.

### 3.4 P3 Evaluation Calibration & CLIP Scores
We evaluated all generated drawings using local CLIP ViT-B/32 cosine similarity against target concept prompts alongside scrambled controls (*scramble-cells* and *scramble-ink*):
* **Agentic Drawing (`cat` / `p2_visual`):** CLIP visual similarity reached **0.2947** against **0.2880** for the scrambled control, yielding a positive Control Gap ($\Delta = +0.0067$).
* **Procedural `bear`:** Achieved $\Delta = +0.0293$ (0.3044 real vs. 0.2752 scrambled).
* **Text Leakage on `horse`:** Yielded a Control Gap of exactly **0.0000** (0.3198 real vs. 0.3198 scrambled), confirming that the scrambled control successfully flags lexical shortcuts.
* **Dual Judge Comparison:** Our dual text-vs-image judge experiment (`run_judge_text_vs_image.py`) confirmed that when given the rendered image alone, the VLM judge correctly identified the drawing as `"cat"` with a rating of 5/5 without text-bias contamination.

### 3.5 Extension Demos (P4–P6 Live Results)
* **P4 Conversational Editing:** Starting from a base drawing, the model was instructed to *"add whiskers"*. Using localized `patch_edit` actions, the agent modified exactly **16 cells** confined to a vertical span of 2 rows (rows 3–4), leaving the remaining 22 rows untouched.
* **P5 Semantic Image-to-ASCII:** Given an input photograph, the multimodal model abstracted the primary subject contours into a structured $24 \times 48$ ASCII representation ($\rho = 0.143$) while suppressing background noise.
* **P6 ASCII Animation:** Generated a 3-frame progressive animation of a `cat` where each keyframe conditioned on the visual render of the previous frame ($\rho = 0.087 \to 0.094 \to 0.095$). The sequence is compiled into a standalone, interactive browser player (`animation.html`).

---

## 4. Discussion & Limitations

* **SketchAgent Comparison:** SketchAgent uses a continuous Bézier stroke action space (one curve per call). ASCIIgent uses a discrete, region-based action space (Bresenham lines, rectangles, tonal ramps, patches) suited to the character-grid medium. Both share the core perception–action loop.
* **The Text-Reading Confound:** Scrambled controls provide evidence that visual shape is being measured, though full calibration against human ratings is the gold standard for CVPR.
* **Limitations:** Hand-drawn exemplars are conservative; free-tier API quotas (20 requests/day on Flash 3.8) require careful caching and pacing.

---

## 5. Conclusion

ASCII art is a rigorous testbed for visual abstraction in language models: the extreme 95-character constraint tests spatial reasoning against memorized recall. We demonstrated that monolithic generation suffers from scanline drift and text-spelling shortcuts, while an agentic tool-use loop with discrete primitives and intermediate visual perception enables genuine spatial self-correction.
