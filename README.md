# Planaura for ArcGIS Pro

> Change detection in satellite imagery using Canadian geospatial foundation models — packaged as ArcGIS Pro Deep Learning Packages (DLPK).

## About Planaura

Planaura is a collection of Canadian geospatial foundation models developed at the **Canada Centre for Mapping and Earth Observation** at **Natural Resources Canada (NRCan)**. It is trained on satellite imagery from Harmonized Landsat and Sentinel (HLS) and Sentinel-2 (S2) sources, covering the Canadian landscape from 2015–2024.

In **bi-temporal mode**, Planaura compares two images of the same location at different dates by generating high-dimensional embeddings (768-dim) using a Vision Transformer (ViT) architecture. Change magnitude is measured using cosine similarity between the before/after embeddings — lower similarity means greater change.

**Resources:**
- 🤗 Hugging Face: https://huggingface.co/NRCan/Planaura-1.0
- 📦 GitHub (upstream): https://github.com/NRCan/planaura/

---

## Prerequisites

- **ArcGIS Pro** 3.1 or later
- **Image Analyst** extension
- ArcGIS deep learning frameworks installed ([instructions](https://pro.arcgis.com/en/pro-app/latest/help/analysis/image-analyst/install-deep-learning-frameworks-for-arcgis.htm))
- Python environment with `arcgis` package (included with ArcGIS Pro)

---

## Model Variants

| Model | Best For | Resolution | Folder |
|-------|----------|------------|--------|
| **Planaura_HLS** | Harmonized Landsat + Sentinel imagery | 30 m | `HLS/` |
| **Planaura_S2** | Sentinel-2 imagery | 10–20 m | `S2/` |

Choose the variant that matches your input imagery source.

---

## Repository Structure

```
├── README.md
├── HLS/
│   ├── Planaura.py           # ArcGIS Python raster function
│   └── Planaura_HLS.emd      # Esri model definition
├── S2/
│   ├── Planaura.py           # ArcGIS Python raster function
│   └── Planaura_S2.emd       # Esri model definition
└── lib/                      # Shared vendored libraries
```

---

## Quick Start

### Step 1 — Build the 12-band composite

The model requires a single **12-band composite raster** stacking your *before* and *after* images in this exact order:

| Bands | Image | Spectral Bands |
|-------|-------|----------------|
| 1–6 | Before | B02 (Blue), B03 (Green), B04 (Red), B8A (NIR), B11 (SWIR1), B12 (SWIR2) |
| 7–12 | After | B02 (Blue), B03 (Green), B04 (Red), B8A (NIR), B11 (SWIR1), B12 (SWIR2) |

In ArcGIS Pro, use the **Composite Bands** geoprocessing tool (Data Management › Raster › Raster Processing › Composite Bands) to stack the before and after images in the correct band order before running the model.

> ⚠️ **Band order matters.** The model will silently produce incorrect results if bands are stacked in the wrong sequence.

### Step 2 — Run the classifier

**Geoprocessing Tool:** Image Analyst › Deep Learning › **Classify Pixels Using Deep Learning**

| Setting | Value |
|---------|-------|
| Input Raster | Your 12-band composite |
| Model Definition | `Planaura_HLS.dlpk` or `Planaura_S2.dlpk` |

**Default Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `padding` | `64` | Tile overlap in pixels to reduce edge artifacts |
| `batch_size` | `1` | Number of tiles processed per batch |
| `No Change Threshold` | `0.8` | Cosine similarity above which pixels are classified as No Change |
| `Low Change Threshold` | `0.6` | Lower bound for Low Change class |
| `Moderate Change Threshold` | `0.4` | Lower bound for Moderate Change class |

Adjust thresholds to tune sensitivity for your landscape and season.

---

## Output Classes

The model outputs a classified raster with the following schema:

| Value | Class | Cosine Similarity |
|------:|-------|-------------------|
| 0 | NoData | N/A |
| 1 | No Change | > 0.80 |
| 2 | Low Change | 0.60 – 0.80 |
| 3 | Moderate Change | 0.40 – 0.60 |
| 4 | High Change | ≤ 0.40 |

---

## Tips & Known Limitations

- **Seasonal variation** can produce false positives — ideally, use images from the same season across years.
- **Cloud and snow cover** in either image will affect embedding quality. Pre-filter or mask cloudy pixels before compositing.
- The model was trained primarily on Canadian landscapes; performance in other regions may vary.
- `batch_size=1` is safe for most machines. Increase only if you have sufficient GPU VRAM.
- For large areas, consider tiling your input or using a mosaic dataset.

---

## Acknowledgments

The Planaura foundation model was developed by the **Canada Centre for Mapping and Earth Observation, Natural Resources Canada (NRCan)**.

This ArcGIS Pro DLPK packaging — including the Python raster functions and Esri model definitions — was developed by **Mohamed Ahmed, Esri Canada** (Education and Research Group), to make Planaura accessible within the ArcGIS Pro deep learning workflow.

For questions about the DLPK packaging, open an issue in this repository.  
For questions about the underlying model, refer to the [upstream Planaura repository](https://github.com/NRCan/planaura/).

---

## License

This DLPK is based on the Planaura model by Natural Resources Canada. See the [original repository](https://github.com/NRCan/planaura/) for full license terms.

---

## Citation

If you use Planaura in your work, please cite the original model via the [NRCan Hugging Face page](https://huggingface.co/NRCan/Planaura-1.0).
