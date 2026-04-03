# 🎨 Advanced Text-to-Image Generation Pipeline
### Internship Project — All 6 Tasks | Built on Stable Diffusion

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Umeshjamra/Stable-diffusion-project-/blob/main/advanced_text_to_image_all_tasks.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers%204.30-yellow.svg)](https://huggingface.co)
[![Diffusers](https://img.shields.io/badge/Diffusers-0.21.0-orange.svg)](https://huggingface.co/docs/diffusers)
[![Gradio](https://img.shields.io/badge/Gradio-4.0-green.svg)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

> ⚠️ **Requires Google Colab with GPU** — Runtime → Change runtime type → **T4 GPU**

---

## 📋 Problem Statement

Extend a Stable Diffusion training project into a **comprehensive, production-style text-to-image pipeline** that integrates:

- Real **CLIP + BERT** text embeddings via HuggingFace Transformers
- **Self-Attention (SAGAN)** and **Cross-Attention** modules inside a GAN
- **LoRA** and **Textual Inversion** fine-tuning for domain adaptation
- **Oxford 102 Flowers** public dataset analysis and statistics
- Multi-model text preprocessing (CLIP tokenisation + BERT embeddings)
- **Conditional GAN (CGAN)** generating geometric shapes from text labels
- All modules wired into a **5-tab interactive Gradio UI**

---

## 📁 Repository Structure

```
Stable-diffusion-project-/
│
├── advanced_text_to_image_all_tasks.ipynb   ← 🔑 MAIN notebook (all 6 tasks)
├── requirements.txt                          ← all pip dependencies
├── README.md                                 ← this file
│
└── dataset/
    ├── oxford102_flowers_dataset.csv         ← Task 4 — 102 classes + statistics
    ├── cgan_shape_dataset.csv                ← Task 6 — 2000 training samples
    ├── text_embeddings_dataset.csv           ← Task 5 — 12 prompts + embeddings
    ├── cgan_training_log.csv                 ← Tasks 1,2,6 — 40-epoch training log
    ├── sd_generation_log.csv                 ← Tasks 1,3 — SD generation metadata
    ├── model_comparison.csv                  ← All tasks — baseline vs advanced
    └── dataset_info.json                     ← dataset description + row counts
```

---

## ✅ Tasks Implemented

| # | Task | Module | Key Output File |
|---|------|--------|----------------|
| 1 | Comprehensive Text-to-Image Pipeline | `ComprehensiveTTIPipeline` | `task1_pipeline_demo.png` |
| 2 | Attention Mechanisms (Self + Cross) | `SelfAttention`, `CrossAttention` | `task2_attention_maps.png` |
| 3 | Fine-Tuning Pre-trained Model | `LoRALinear`, `TextualInversionEmbedding` | `task3_finetuning.png` |
| 4 | Dataset Loading & Analysis | `DatasetAnalyser` | `task4_dataset_analysis.png` |
| 5 | Text Preprocessing & Embeddings | `TextEmbeddingPipeline` | `task5_text_embeddings.png` |
| 6 | Conditional GAN — Shape Generation | `CGANGenerator`, `CGANDiscriminator` | `task6_cgan_results.png` |

---

## 🏗️ Architecture

### Full Pipeline Flow (Task 1)

```
Text Prompt
     │
     ▼
[Task 5] TextEmbeddingPipeline
     ├─ CLIP  (openai/clip-vit-base-patch32)  →  (batch, 77, 512)
     └─ BERT  (bert-base-uncased)             →  (batch, 768)  [CLS token]
     │
     ▼
[Task 1] ComprehensiveTTIPipeline.run(prompt)
     ├─ Step 1: preprocess()   → clean text + word/char stats
     ├─ Step 2: encode()       → CLIP embeddings
     ├─ Step 3: generate_cgan()→ [Task 6] CGAN → 32×32 shape (instant)
     └─ Step 4: generate_sd()  → StableDiffusionPipeline → 512×512 (optional)
```

### Task 2 — Attention Modules

**SelfAttention (SAGAN-style, Zhang et al. 2018):**
```python
q = Wq(x).view(B, C//8, HW).T    # (B, HW, C//8)
k = Wk(x).view(B, C//8, HW)      # (B, C//8, HW)
attn = softmax(q @ k)             # (B, HW, HW)  — every pixel attends to all others
out  = gamma * (v @ attn.T) + x   # learnable residual, gamma starts at 0
```

**CrossAttention (text → image, same design as SD's U-Net):**
```python
q = Wq(img_feat)    # (B, HW, C)   — image queries
k = Wk(text_emb)    # (B, T,  C)   — text keys
v = Wv(text_emb)    # (B, T,  C)   — text values
attn = softmax(q @ k.T / sqrt(d_head))
out  = attn @ v + img_feat          # multi-head, 4 heads, residual
```

### Task 6 — CGAN Architecture

```
CGANGenerator(z, label):
  z [100] + label_emb(label) [16]  →  concat [116]
  → Linear(116 → 256×8×8)  →  reshape  →  (B, 256, 8, 8)
  → ConvTranspose2d  8→16  (128ch) → BN → ReLU
  → ConvTranspose2d  16→32  (64ch) → BN → ReLU
  → SelfAttention(64ch)             ← Task 2 integration at 32×32
  → Conv2d(64→1, 3×3) → Tanh       →  (B, 1, 32, 32)

CGANDiscriminator(img, label):
  label_emb(label) → project → (B, 1, 32, 32) spatial map
  concat [image, label_map]  →  (B, 2, 32, 32)
  → Conv2d(2→64, stride 2) → LeakyReLU → Dropout2d(0.25)
  → Conv2d(64→128, stride 2) → BN → LeakyReLU → Flatten
  → Linear → Sigmoid  →  P(real | label)
```

### Task 3 — Fine-Tuning

**LoRA (LoRALinear):**
```python
# W_eff = W_frozen + (alpha / rank) * (x @ A @ B)
# A ∈ R^(d_in × rank),  B ∈ R^(rank × d_out),  B initialised to zero
# rank=4, d=768 → 6,144 trainable vs 589,824 frozen  (≈ 1% of full)
```

**Textual Inversion (TextualInversionEmbedding):**
```python
# Adds one learnable vector: token_embedding (1, 768)
# Token name: <domain-token>   Init: N(0, 0.01)
# All model weights frozen — only this embedding is trained
```

---

## 📊 Dataset

**Google Drive:** [📁 Dataset Folder](https://drive.google.com/drive/folders/YOUR_FOLDER_ID)

All data is **original** — generated directly from the notebook's classes, parameters, and training runs.

| File | Rows | Task | Description |
|------|------|------|-------------|
| `oxford102_flowers_dataset.csv` | 102 | Task 4 | All 102 Oxford Flowers classes with n_images, avg caption words, resolution, dominant colour, split |
| `cgan_shape_dataset.csv` | 2,000 | Task 6 | ShapeDataset: 500 samples × 4 classes (circle, square, triangle, star) with geometry params |
| `text_embeddings_dataset.csv` | 12 | Task 5 | 12 sample prompts with CLIP token counts, word stats, 16-dim embedding vectors |
| `cgan_training_log.csv` | 40 | T1,2,6 | 40-epoch CGAN training: D_loss, G_loss, D(x), D(G(z)), lr, batch_size per epoch |
| `sd_generation_log.csv` | 10 | T1,3 | SD generation runs: model, scheduler, steps, CFG scale, seed, width, height, time |
| `model_comparison.csv` | 6 | All | Baseline → CGAN → CGAN+Attn → SD → LoRA → Textual Inversion benchmark |

**Total: 2,170 rows across 6 files**

---

## 📈 Results & Visualisations

All plots are auto-saved when the notebook runs:

### Task 1 — Comprehensive Pipeline Demo
`task1_pipeline_demo.png` — 4-panel: text prompts mapped to CGAN-generated shapes (circle, square, triangle, star)

### Task 2 — Attention Maps
`task2_attention_maps.png` — 3-panel: input feature map · self-attention from centre pixel · after cross-attention with text

### Task 3 — Fine-Tuning Concepts
`task3_finetuning.png` — 6-panel: LoRA vs full param count (log scale) · textual inversion embedding space · simulated loss curves · domain prompt comparison tables

### Task 4 — Oxford 102 Flowers Analysis
`task4_dataset_analysis.png` — 5-panel: top-20 class distribution · summary stats table · caption length histogram · resolution scatter · top-8 class pie chart

`task4_samples.png` — 6 sample flower images with class labels

### Task 5 — Text Embeddings
`task5_text_embeddings.png` — CLIP token count per prompt (vs max 77) + cosine similarity heatmap

### Task 6 — CGAN Results
`task6_cgan_results.png` — Training loss curves · 4×4 generated shapes grid · one clear sample per class (circle, square, triangle, star)

### Summary Dashboard
`summary_dashboard.png` — Task completion overview + model comparison bar chart

---

## 🔬 Model Comparison

| Model | Task | Architecture | Output | Inference | VRAM | Text Alignment |
|-------|------|--------------|--------|-----------|------|----------------|
| DCGAN baseline | T1 | GAN | 32×32 | ~1 ms | ~50 MB | None |
| CGAN | T6 | Conditional GAN | 32×32 | ~1 ms | ~80 MB | Class label |
| CGAN + SelfAttn | T2+T6 | CGAN + SelfAttention | 32×32 | ~2 ms | ~100 MB | High (class) |
| SD 1.5 (base) | T1 | LDM + CLIP | 512×512 | ~12 s | 4.5 GB | High (CLIP) |
| SD 1.5 + LoRA | T3 | LDM + LoRA (r=4) | 512×512 | ~13 s | 4.8 GB | High |
| SD 1.5 + TI | T3 | LDM + TextualInversion | 512×512 | ~12 s | 4.5 GB | Very High |

**Key finding:** CGAN + SelfAttention is ~6,000× faster than SD 1.5 with ~45× less VRAM. LoRA fine-tunes SD with only ~1% trainable parameters vs full fine-tuning.

---

## 🖥️ Gradio UI (5 Tabs)

| Tab | Task | Features |
|-----|------|---------|
| 🚀 Stable Diffusion | T1 | Model init, prompt, negative prompt, width/height, steps, CFG, scheduler, seed, save |
| 🔷 CGAN Shapes | T6 | Generate circle/square/triangle/star from text prompt |
| 📝 Text Embeddings | T5 | Multi-prompt word stats + CLIP analysis |
| 🖼️ Examples & Gallery | T1 | Pre-loaded example prompts + recent generation gallery |
| 📚 About the Project | All | Architecture table, key technologies, task modules |

---

## 🚀 Quick Start

### Option A — Google Colab (recommended)
1. Click the **Open in Colab** badge above
2. Runtime → Change runtime type → **T4 GPU**
3. Runtime → **Run all** (~10–15 min first run for model downloads)
4. Gradio UI appears with a public share link

### Option B — Local
```bash
git clone https://github.com/Umeshjamra/Stable-diffusion-project-.git
cd Stable-diffusion-project-
pip install -r requirements.txt
jupyter notebook advanced_text_to_image_all_tasks.ipynb
```

---

## 📦 Requirements

```
torch==2.0.1+cu118
torchvision==0.15.2+cu118
diffusers==0.21.0
transformers==4.30.2
accelerate==0.20.3
safetensors==0.3.1
xformers==0.0.20
Pillow==9.5.0
numpy==1.24.4
matplotlib==3.7.2
seaborn
gradio==4.0.0
datasets
ftfy
regex
tqdm
scipy
```

Install: `pip install -r requirements.txt`

---

## 📚 References

1. Rombach et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022.
2. Zhang et al. (2018). *Self-Attention Generative Adversarial Networks.* ICML 2019.
3. Gal et al. (2022). *An Image is Worth One Word: Personalizing Text-to-Image via Textual Inversion.* ICLR 2023.
4. Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
5. Mirza & Osindero (2014). *Conditional Generative Adversarial Nets.* arXiv 2014.
6. Radford et al. (2021). *Learning Transferable Visual Models from Natural Language (CLIP).* ICML 2021.
7. Devlin et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL 2019.
8. Nilsback & Zisserman (2008). *Automated Flower Classification over a Large Number of Classes.* ICVGIP 2008.

---

## 👤 Author

**Umesh Jamra** | Internship 2025
**GitHub:** [github.com/Umeshjamra/Stable-diffusion-project-](https://github.com/Umeshjamra/Stable-diffusion-project-)
