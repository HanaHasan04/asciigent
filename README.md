# ASCIIgent: Agentic ASCII Art Generation

**Authors:** Hana Hasan, Amit Arad, Roei Levi, Tom Shur

ASCIIgent is a training-free framework for generating, refining, editing, and animating ASCII art ($24 \times 48$ printable character grids) using multimodal Large Language Models (VLMs). Instead of emitting an entire character grid in a single left-to-right pass—which suffers from scanline drift and lexical leakage—ASCIIgent executes a closed-loop **draft $\to$ render $\to$ perceive $\to$ revise** cycle over a discrete spatial toolset (`AsciiCanvas`).

---

## Repository Structure

```text
asciigent/
├── asciigent/                   # Core Python package (flat layout — no src/ wrapper)
│   ├── actions.py               # Discrete action space & AsciiCanvas (Bresenham lines, rects, tonal ramps, patch_edit)
│   ├── agent.py                 # Multi-turn active visual perception loop (visual, text_only, no_feedback)
│   ├── contact_sheet.py         # Contact sheet generator
│   ├── env.py                   # Automatic .env loader
│   ├── grid.py                  # 24×48 AsciiGrid, 2-digit coordinate rulers, scrambling controls
│   ├── judge.py                 # VLM Model-as-Judge with scrambled control gap
│   ├── llm.py                   # Disk-cached, zero-budget-enforced multimodal LLM client
│   ├── llm_types.py             # Usage tracking, LLMResponse, price tables
│   ├── metrics.py               # Structural metrics (ink ratio, coverage, Shannon entropy, spells_name)
│   ├── metrics_vision.py        # CLIP ViT-B/32 cosine similarity on rendered bitmaps
│   ├── providers.py             # Provider presets (gemini, openrouter, groq, local)
│   ├── render.py                # Deterministic cell-by-cell monospace bitmap renderer
│   ├── store.py                 # Per-run artifact logging and JSONL index persistence
│   ├── experiments/
│   │   ├── p1_text2ascii.py     # P1: One-shot baseline sweep (zero-shot, few-shot, procedural CoT)
│   │   ├── p2_agent.py          # P2: Multi-turn agentic loop & feedback mode ablation
│   │   ├── p3_judge.py          # P3: Calibrated VLM judge with scrambled controls
│   │   ├── p4_edit.py           # P4: Localized conversational editing via patch_edit
│   │   ├── p5_image2ascii.py    # P5: Semantic photograph-to-ASCII abstraction
│   │   └── p6_animate.py        # P6: Delta-patched keyframe ASCII animation
│   └── prompts/
│       ├── agent.py             # System, draft, and visual critique prompts
│       ├── judge.py             # Unprompted identify and 1–5 rating prompts
│       └── text2ascii.py        # Shared 24×48 canvas contract for P1 baselines
├── data/
│   ├── categories.json          # Seen, held-out, and exemplar-only splits
│   └── exemplars/               # Hand-crafted exemplars (fish, house, sailboat, tree)
├── report/
│   ├── main.tex                 # 3-page IEEE/CVPR-style LaTeX report
│   ├── main.pdf                 # Compiled 3-page PDF report with all figures
│   ├── references.bib           # BibTeX bibliography
│   └── figures/                 # High-contrast figures used in the report
├── runs/                        # Saved experimental runs, turn renders, and animations
├── scripts/
│   ├── analyze_p1.py            # Quantitative summary tables for P1
│   ├── evaluate_clip.py         # CLIP ViT-B/32 evaluation with scrambled controls
│   ├── generate_report_figures.py # Rebuilds Figures 2–5 from runs/
│   ├── make_grid_contact_sheet.py # Rebuilds Figure 1 (12-cell P1 contact sheet)
│   └── run_judge_text_vs_image.py # Image-vs-text VLM judge comparison
└── tests/                       # 87 unit and integration tests
```

---

## Quickstart & Reproducibility

Because `asciigent/` is located directly at the repository root, all commands run straight from the root directory without needing `PYTHONPATH`.

### 1. Install Dependencies & Run Unit Tests
```powershell
pip install -r requirements.txt
python -m pytest tests/ -q
```

### 2. Configure API Key (Optional for Cached Runs)
All completed runs and LLM responses are cached. To run new live queries:
```powershell
cp .env.example .env
# Add your free GEMINI_API_KEY or OPENROUTER_API_KEY inside .env
```

### 3. Reproduce Experiments (P1–P6)
```powershell
# P1: One-Shot Baselines (Zero-Shot, Few-Shot, Procedural CoT)
python -m asciigent.experiments.p1_text2ascii --split seen --samples 1
python scripts/analyze_p1.py runs/index.jsonl

# P2: Agentic Self-Perception Loop & Feedback Ablation
python -m asciigent.experiments.p2_agent --category cat --max-turns 3 --feedback visual
python -m asciigent.experiments.p2_agent --category cat --max-turns 3 --feedback text_only
python -m asciigent.experiments.p2_agent --category cat --max-turns 3 --feedback no_feedback

# P3: Calibrated VLM Judge & CLIP Evaluation
python -m asciigent.experiments.p3_judge --runs-dir runs
python scripts/evaluate_clip.py runs/index.jsonl

# P4: Localized Conversational Editing
python -m asciigent.experiments.p4_edit --drawing runs/p4_edit/before.txt --instruction "add a pointy party hat with a star on top of the head"

# P5: Semantic Image-to-ASCII
python -m asciigent.experiments.p5_image2ascii --image runs/p5_image2ascii/input_photo.png

# P6: Delta-Patched Keyframe Animation
python -m asciigent.experiments.p6_animate --category cat --frames 3
```

### 4. Regenerate Report Figures
```powershell
python scripts/make_grid_contact_sheet.py
python scripts/generate_report_figures.py
```
