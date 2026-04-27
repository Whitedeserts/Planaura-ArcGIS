# Planaura HLS for ArcGIS Pro

This repository provides reference documentation for the **Planaura HLS Deep Learning Package (DLPK)** for use with **ArcGIS Pro**.

## About Planaura

Planaura is a deep learning model for change detection in satellite imagery. For more information about the model, training, and additional resources:

- **Hugging Face:** https://huggingface.co/NRCan/Planaura
- **GitHub:** https://github.com/NRCan/planaura/

## Repository Structure

```
├── Planaura.py          # ArcGIS Python raster function
├── Planaura_HLS.emd     # Esri model definition
└── lib/                 # Vendored support libraries
```

Use it with:

**Image Analyst → Deep Learning → Classify Pixels Using Deep Learning**

## Purpose

This DLPK performs **change classification** between two dates and outputs a **single-band classified raster**.

## Input Requirements

Provide a **12-band composite raster** with this exact layout:

### Bands 1–6 = before image
- B02
- B03
- B04
- B8A
- B11
- B12

### Bands 7–12 = after image
- B02
- B03
- B04
- B8A
- B11
- B12

This **HLS Planaura DLPK** is intended for **HLS imagery**.

## ArcGIS Tool Settings

- **Tool**: Classify Pixels Using Deep Learning
- **Model Definition**: `Planaura_HLS.dlpk`

Typical arguments:

```text
padding=64; batch_size=1; use_f16=0
```

## Output Classes

The output raster uses these class values:

| Value | Label | Meaning |
|------:|-------|---------|
| 0 | NoData | invalid / missing data |
| 1 | No Change | cosine > 0.80 |
| 2 | Low Change | cosine > 0.60 and <= 0.80 |
| 3 | Moderate Change | cosine > 0.40 and <= 0.60 |
| 4 | High Change | cosine <= 0.40 |

## Recommended Symbology

- 0 = transparent or light gray
- 1 = dark green
- 2 = yellow-green
- 3 = orange
- 4 = red


## Quick Usage Note

> Use this DLPK with **Classify Pixels Using Deep Learning** on a 12-band composite raster where bands 1–6 are the before image and bands 7–12 are the after image. 
Output classes are 0 = no-data, 1 = no change, 2 = low change, 3 = moderate change, and 4 = high change.
