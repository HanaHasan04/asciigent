# ASCIIgent: Training-Free Agentic ASCII Art Generation via Active Visual Perception

> **Modern Computer Vision — Final Project & CVPR Research Prototype**  
> Hana Hasan, Amit Arad, Roei Levi, Tom Shur

> **Import audit — 2026-10-25:** This is a research prototype, not a completed
> experimental submission. P2 currently implements visual feedback only; its
> no-feedback and text-only critique controls remain to be implemented. The
> report contains numerical claims that cannot be verified from the supplied
> run artifacts. Treat both report versions as unverified drafts. The ZIP did
> not include Git history or individual drawing directories. CLIP additionally
> requires PyTorch, the OpenAI CLIP package, and model weights; these are not
> included in `requirements.txt`. Original imported files and the previous
> README are preserved under `.update-backup-20261025/`.

ASCIIgent investigates whether Large Language Models (LLMs) can draw with a 95-character discrete alphabet through generalizable spatial reasoning rather than memorized recall, and whether a training-free agentic loop—**draft $\to$ render $\to$ visually perceive $\to$ revise**—overcomes the fundamental *autoregressive scanline blindness* of standard models.

---

## 📑 Repository Structure & Component Guide

```
asciigent/
├── data/
│   ├── categories.json          # Seen, held-out, and exemplar-only splits
│   └── exemplars/               # Hand-crafted exemplars (fish, house, sailboat, tree)
├── docs/
│   ├── p1_hypotheses.md         # Pre-registered predictions and kill criteria
│   └── cvpr_blueprint.md        # Comprehensive CVPR research proposal & blueprint
├── report/
│   ├── main.tex                 # 2-3 page LaTeX paper (MCV / CVPR format)
│   ├── asciigent_report.md      # Full markdown version of the final paper
│   └── references.bib           # Complete BibTeX bibliography
├── runs/                        # Run outputs (drawings, logs, contact sheets)
│   ├── index.jsonl              # Summary index of all runs
│   └── p1_contact_sheet.png     # Rendered visual contact sheet
├── scripts/
│   ├── analyze_p1.py            # Summary tables (split × arm means ± SD)
│   ├── check_provider.py        # 3-call API smoke test & quota diagnostic
│   ├── evaluate_clip.py         # Evaluates drawings using CLIP + control gaps
│   └── smoke_render.py          # Monospace rendering verification
├── src/asciigent/
│   ├── actions.py               # Discrete action space & AsciiCanvas (Bresenham, tonal ramp, patch)
│   ├── agent.py                 # Multi-turn SketchAgent-style active perception loop
│   ├── contact_sheet.py         # Contact sheet generator (category rows × arm columns)
│   ├── env.py                   # Automatic .env parser
│   ├── grid.py                  # 24×48 AsciiGrid, damage reports, scrambling controls
│   ├── judge.py                 # VLM Model-as-judge with control gap check
│   ├── llm.py                   # Cached, zero-budget-enforced LLM client
│   ├── llm_types.py             # Usage tracking, LLMResponse, price tables
│   ├── metrics.py               # Structural metrics (ink ratio, entropy, spells-name)
│   ├── metrics_vision.py        # CLIP cosine similarity on rendered bitmaps
│   ├── providers.py             # Provider presets (gemini, openrouter, groq, local)
│   ├── render.py                # Deterministic cell-by-cell monospace to bitmap renderer
│   ├── store.py                 # Per-run artifact logging and index persistence
│   ├── experiments/
│   │   ├── p1_text2ascii.py     # P1 sweep: seen/held-out split × 3 prompt arms
│   │   ├── p2_agent.py          # P2 agentic sweep: multi-turn drawing with visual feedback
│   │   └── p3_judge.py          # P3 judge sweep: model-as-judge scoring
│   └── prompts/
│       ├── agent.py             # Prompts for agent actions and visual critique
│       ├── judge.py             # Separate identify and rate prompts
│       └── text2ascii.py        # Shared canvas contract for zero-shot, few-shot, procedural
└── tests/                       # 84 offline tests verified after import
```

---

## ⚡ Step-by-Step Instructions: From Setup to Submission

### Step 1: Environment Setup
Ensure Python 3.10+ is installed with the required dependencies:
```bash
pip install -r requirements.txt
cp .env.example .env
```
Paste your free API key into `.env` (e.g. `GEMINI_API_KEY=...` or `OPENROUTER_API_KEY=...`). real environment variables will also override this.

### Step 2: Verify Provider & Run Tests
Verify your API credentials and run the offline test suite:
```bash
# Smoke test your provider
python scripts/check_provider.py

# Run the offline tests (84 passed during the 2026-10-25 import audit)
python -m pytest tests/ -q
```

### Step 3: Run the P1 Baseline Sweep (Monolithic Prompting)
Test monolithic zero-shot, few-shot, and procedural prompting across seen/held-out categories:
```bash
# Dry run to preview prompts without sending API calls:
PYTHONPATH=src python -m asciigent.experiments.p1_text2ascii --dry-run

# Run seen categories (1 sample each):
PYTHONPATH=src python -m asciigent.experiments.p1_text2ascii --split seen --samples 1

# Analyze results:
python scripts/analyze_p1.py runs/index.jsonl
```

### Step 4: Run the P2 Agentic Loop (Active Perception)
Execute the multi-turn agent that plans, draws with discrete tools, renders intermediate states, and visually self-corrects:
```bash
# Dry run to validate configuration:
PYTHONPATH=src python -m asciigent.experiments.p2_agent --dry-run

# Run an agentic session on 'cat' with up to 4 turns of visual feedback:
PYTHONPATH=src python -m asciigent.experiments.p2_agent --category cat --max-turns 4
```
Each intermediate turn will be saved in `runs/<run_id>/turn_*.png` alongside the final drawing.

### Step 5: Evaluate Drawings with Calibrated Metrics (P3)
Score drawings using both VLM-as-Judge and multimodal CLIP embeddings with structure-destroying controls:
```bash
# 1. Run VLM Model-as-Judge over all generated drawings:
PYTHONPATH=src python -m asciigent.experiments.p3_judge --runs-dir runs

# 2. Compute CLIP visual cosine similarity and Control Gap:
python scripts/evaluate_clip.py runs/index.jsonl

# Or evaluate a single drawing directly:
python scripts/evaluate_clip.py --drawing data/exemplars/fish.txt --category fish
```

### Step 6: Review & Submit Final Report
The report drafts are in the `report/` folder and require verified experimental results before submission:
- **Markdown Report:** [`report/asciigent_report.md`](report/asciigent_report.md) — unverified draft.
- **LaTeX Paper (2-3 Pages):** [`report/main.tex`](report/main.tex) and [`report/references.bib`](report/references.bib) — compile with standard `pdflatex` or import into Overleaf.

---

## 🔒 Hard Invariants & Zero-Cost Policy

1. **This project costs nothing.** Metered providers raise `PaidProviderRefused` unless `allow_paid=True` is explicitly passed.
2. **Every request is cached on disk under `.cache/llm/`.** Re-running sweeps only calls missing or modified cells.
3. **Format contract:** All drawings are strictly $24 \times 48$ matrices of printable ASCII. Any repairs made are logged in a `ParseReport`.
4. **Scrambled controls:** The visual judge defaults to `scramble_cells`; add `--controls cells ink` for both controls. CLIP computes both. The ink scramble preserves silhouette, so report its gap separately when interpreting sensitivity to shape. These controls provide evidence about the metric, not proof that text bias is absent.
