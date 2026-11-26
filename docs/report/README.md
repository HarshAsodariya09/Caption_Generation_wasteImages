# 📄 CaptionGene LaTeX Technical Report Bundle

Everything required to compile your complete PDF report is self-contained inside this folder (`docs/report/`):

- **`main.tex`** — Complete IEEE/Overleaf-ready technical report covering Goals, TACO Dataset & Schema, Kaggle GPU LoRA Fine-Tuning & Obstacle Fixes, Unconstrained vs. Taxonomy Before/After Results, GenEval & DPG-Bench Comparisons, Error Analysis, and the Extra Conveyor Belt Sorting Simulator.
- **`figures/`**:
  - `fig1_dataset_distribution.png` — Object Category (n=888) & Recycling Bin Distribution
  - `fig2_blip2_loss_curve.png` — Real Kaggle Tesla T4 LoRA Loss Convergence ($1.9891 \to 0.1057$)
  - `fig3_before_after_metrics.png` — Before vs. After Progression (GTA, Bin Acc, Stream %, DPG-Bench, GenEval)
  - `fig4_qualitative_showcase.png` — Real TACO Visual Showcase (`taco_000000`, `taco_000001`, `taco_000004`)

## 1-Click PDF Compilation on Overleaf (Free, Online)
Since no local TeX compiler is installed on this Windows machine:
1. Zip the `docs/report/` folder (`main.tex` + `figures/`).
2. Go to [Overleaf](https://www.overleaf.com/) $\to$ **New Project** $\to$ **Upload Project** and upload the `.zip`.
3. Click **Recompile** (`pdfLaTeX`), then **Download PDF**!
