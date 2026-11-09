# MM-IMDb Multimodal Movie Genre Classification

Predicting movie genres (23 classes, multi-label) from **posters** and **plot summaries**, and comparing unimodal models with **early fusion**, **late fusion** and **early fusion with PCA-balanced modalities**.

Author: Sebin Chooravadil Benny

---

## 1. Overview

**Research question:** Does combining visual (poster) and textual (plot) information improve genre prediction compared with either modality alone?

**Dataset:** MM-IMDb (Arevalo et al., 2017): 25,959 movies, each with a poster, a plot summary and 1 or more of 23 genre labels (on average 2.48 genres per movie). The official split is used: 15,552 train / 2,608 dev / 7,799 test.

**Features used:** the pre-extracted features provided with the dataset.
- Poster: VGG-16 features (4,096 dimensions)
- Plot: averaged word2vec embeddings (300 dimensions)

## 2. Results (test set, mean ± std over 5 random seeds)

| Model | micro-F1 | macro-F1 | samples-F1 | mAP |
|---|---|---|---|---|
| Poster only | 0.454 ± 0.006 | 0.340 ± 0.003 | 0.450 ± 0.006 | 0.306 ± 0.002 |
| Plot only | 0.607 ± 0.003 | 0.528 ± 0.002 | 0.601 ± 0.003 | 0.523 ± 0.001 |
| Early fusion | 0.599 ± 0.003 | 0.520 ± 0.003 | 0.590 ± 0.002 | 0.517 ± 0.002 |
| Late fusion (weight tuned per seed) | 0.626 ± 0.002 | 0.558 ± 0.001 | 0.616 ± 0.002 | 0.562 ± 0.001 |
| **Early fusion + PCA** | **0.627 ± 0.002** | **0.559 ± 0.003** | **0.619 ± 0.002** | **0.567 ± 0.001** |

**Key findings**
- Fusion improves macro-F1 over the best unimodal model (plot) by about 0.03, roughly 10× the seed-to-seed standard deviation.
- Plain early fusion underperforms plot-only. The results suggest that the high dimensionality of the 4,096-d poster features contributed to this. Reducing them to 300-d with PCA (fitted on the training set) gave the highest mean macro-F1 and the fewest parameters among the fusion models; late fusion was close (0.558).
- Late fusion and early fusion + PCA perform comparably; their difference is within one standard deviation.
- Posters help most for visually distinctive genres (Animation +0.27 AP, Film-Noir +0.17 AP).

## 3. Repository structure

```
├── README.md
├── requirements.txt
├── mmimdb_genre_classification.ipynb   # full pipeline: data → models → evaluation
├── figures/                             # all figures used in the report
│   ├── genre_distribution.png
│   ├── labels_per_movie.png
│   ├── loss_Poster.png, loss_Plot.png, loss_Early_fusion.png, loss_Early_fusion_PCA.png
│   ├── late_fusion_alpha.png
│   ├── model_comparison.png
│   ├── per_genre_AP.png
│   └── examples.png
└── results/
    ├── results_5seeds.csv
    ├── per_genre_AP.csv
    ├── efficiency.csv
    └── prediction_bias.csv
```

## 4. How to reproduce

The dataset (about 4 GB) is **not included** in this repository because of its size and because the posters are copyrighted.

**Option A: Kaggle (recommended, no download needed)**
1. Sign in at kaggle.com and create a new notebook (**Create → Notebook**).
2. Upload `mmimdb_genre_classification.ipynb` (**File → Import Notebook**).
3. In the right panel click **Add Input**, search for `mmimdb` and add the dataset by **johnarevalo**: https://www.kaggle.com/datasets/johnarevalo/mmimdb
4. Optional: set **Settings → Accelerator → GPU**. The code also runs on CPU, more slowly.
5. Click **Run All**. The full notebook takes about 5 minutes on a GPU.

**Option B: Local machine**
1. Download the dataset from the Kaggle link above. You need `multimodal_imdb.hdf5` and `metadata.npy`.
2. Install the dependencies: `pip install -r requirements.txt`
3. In the notebook, change the `path` variable in Section 1 to the folder containing the two files.
4. Run all cells in order.

**Reproducibility notes**
- Random seeds 0–4 are used for all repeated runs (`torch.manual_seed`, `np.random.seed`).
- Scalers, PCA, fusion weight α and per-genre thresholds are all fitted on the **train or dev** split only; the test set is used once, for final evaluation.

## 5. Method summary

| Component | Setting |
|---|---|
| Classifier | MLP: Linear → ReLU → Dropout(0.5) → Linear (23 outputs), hidden size 512 |
| Loss | `BCEWithLogitsLoss` with per-genre `pos_weight` (capped at 10) for class imbalance |
| Optimiser | Adam, learning rate 1e-4, weight decay 1e-3, batch size 256, 40 epochs |
| Checkpoint selection | Trained for 40 epochs and restored the checkpoint with the lowest validation loss |
| Thresholds | Tuned per genre on the dev set (0.05–0.90) to maximise F1 |
| Preprocessing | Standardisation (fitted on train); PCA-300 on poster features for the balanced early-fusion model |
| Late fusion | `α · P_poster + (1 − α) · P_plot`, α chosen per seed by dev macro mAP |
| Metrics | micro-F1, macro-F1, samples-F1, macro mAP, per-genre AP |

## 6. Ethics and limitations

- MM-IMDb reflects MovieLens's film selection, not a random sample of world cinema.
- Genres are heavily imbalanced (Drama 53.8% vs Film-Noir 1.3%); rare genres score much lower, and Drama is over-predicted (69% predicted vs 53% true on test).
- Pretrained features may carry biases from their original training data.
- Posters may depict identifiable actors and are copyrighted; no facial recognition was performed and posters are not redistributed here.

## 7. References

Arevalo, J., Solorio, T., Montes-y-Gómez, M. and González, F.A. (2017) 'Gated multimodal units for information fusion', *ICLR 2017 Workshop*. arXiv:1702.01992.

Dataset: https://www.kaggle.com/datasets/johnarevalo/mmimdb
