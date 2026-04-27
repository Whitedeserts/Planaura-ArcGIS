# Planaura HLS for ArcGIS Pro

This repository provides the **Planaura HLS Deep Learning Package (DLPK)** for change detection in satellite imagery using **ArcGIS Pro**.

## About Planaura

Planaura is a collection of Canadian geospatial foundation models developed at the **Canada Centre for Mapping and Earth Observation** at **Natural Resources Canada (NRCan)**. The model is trained on satellite imagery from Harmonized Landsat and Sentinel (HLS) and Sentinel-2 (S2) sources, covering Canada from 2015-2024.

### Model Variants

| Model | Best For | Resolution |
|-------|----------|------------|
| **Planaura_HLS** | HLS imagery | 30m |
| **Planaura_S2** | Sentinel-2 imagery | 10-20m |

### Resources

- **Hugging Face:** https://huggingface.co/NRCan/Planaura-1.0
- **GitHub:** https://github.com/NRCan/planaura/

## Repository Structure

```
├── Planaura.py          # ArcGIS Python raster function
├── Planaura_HLS.emd     # Esri model definition
└── lib/                 # Vendored support libraries
```

## Usage in ArcGIS Pro

**Tool:** Image Analyst > Deep Learning > Classify Pixels Using Deep Learning

**Model Definition:** `Planaura_HLS.dlpk`

**Arguments:**
```
padding=64; batch_size=1; use_f16=0
```

## Input Requirements

Provide a **12-band composite raster** with this exact band order:

| Bands | Image | Spectral Band |
|-------|-------|---------------|
| 1-6 | Before | B02 (Blue), B03 (Green), B04 (Red), B8A (NIR 865nm), B11 (SWIR 1610nm), B12 (SWIR 2190nm) |
| 7-12 | After | B02 (Blue), B03 (Green), B04 (Red), B8A (NIR 865nm), B11 (SWIR 1610nm), B12 (SWIR 2190nm) |

## Output Classes

The model outputs a classified raster based on cosine similarity between the before/after image embeddings:

| Value | Class | Cosine Similarity | Recommended Color |
|------:|-------|-------------------|-------------------|
| 0 | NoData | N/A | Transparent |
| 1 | No Change | > 0.80 | Dark Green |
| 2 | Low Change | 0.60 - 0.80 | Yellow-Green |
| 3 | Moderate Change | 0.40 - 0.60 | Orange |
| 4 | High Change | <= 0.40 | Red |

## How It Works

Planaura uses a Vision Transformer (ViT) architecture to generate high-dimensional embeddings (768-dim) for each image. In bi-temporal mode, it compares embeddings from two dates using cosine similarity to detect changes. Lower cosine values indicate greater change between the images.

### Model Architecture

- **Image Size:** 512 x 512
- **Patch Size:** 16
- **Embedding Dimension:** 768
- **Encoder Depth:** 12 layers
- **Decoder Depth:** 8 layers

## License

This DLPK is based on the Planaura model by Natural Resources Canada. See the [original repository](https://github.com/NRCan/planaura/) for license details.
