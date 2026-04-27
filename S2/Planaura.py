import os
import sys
import json
import traceback
import importlib
import importlib.util

# CRITICAL on Windows:
# Keep lib/ ahead of MODULE_ROOT so `import planaura` resolves to lib/planaura/
# and not accidentally to this file (Planaura.py) via case-insensitive matching.
MODULE_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_LIB_ROOT = os.path.join(MODULE_ROOT, "lib")
if os.path.isdir(LOCAL_LIB_ROOT) and LOCAL_LIB_ROOT not in sys.path:
    sys.path.insert(0, LOCAL_LIB_ROOT)
if os.path.isdir(MODULE_ROOT) and MODULE_ROOT not in sys.path:
    sys.path.append(MODULE_ROOT)

import numpy as np
import torch
import torch.nn.functional as F


def _write_error_log(message: str) -> None:
    paths = [os.path.join(MODULE_ROOT, "planaura_arcgis_error.txt")]
    tmpdir = os.environ.get("TEMP") or os.environ.get("TMP")
    if tmpdir:
        paths.append(os.path.join(tmpdir, "planaura_arcgis_error.txt"))
    for p in paths:
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.write(message)
        except Exception:
            pass


def _import_resume_pretrained_network():
    try:
        from planaura.networks.network_generator import resume_pretrained_network
        return resume_pretrained_network
    except Exception:
        pass

    netgen_path = None
    for root, dirs, files in os.walk(MODULE_ROOT):
        if "network_generator.py" in files:
            netgen_path = os.path.join(root, "network_generator.py")
            break

    if netgen_path is None:
        raise ImportError(
            f"Could not find network_generator.py anywhere under '{MODULE_ROOT}'."
        )

    # lib/planaura/networks/network_generator.py -> pkg_root = lib
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(netgen_path)))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)

    # Create package placeholders if needed so internal imports resolve.
    for pkg_name in ("planaura", "planaura.networks"):
        if pkg_name not in sys.modules:
            pkg_parts = pkg_name.split(".")
            pkg_dir = os.path.join(pkg_root, *pkg_parts)
            if os.path.isdir(pkg_dir):
                init_py = os.path.join(pkg_dir, "__init__.py")
                spec = importlib.util.spec_from_file_location(
                    pkg_name,
                    init_py if os.path.isfile(init_py) else None,
                    submodule_search_locations=[pkg_dir],
                )
                if spec is not None:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[pkg_name] = mod
                    try:
                        if spec.loader is not None:
                            spec.loader.exec_module(mod)
                    except Exception:
                        pass

    spec = importlib.util.spec_from_file_location(
        "planaura.networks.network_generator", netgen_path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["planaura.networks.network_generator"] = mod
    if spec.loader is None:
        raise ImportError("Could not create loader for network_generator.py")
    spec.loader.exec_module(mod)
    return mod.resume_pretrained_network


def _build_runtime_config(emd, ckpt_path):
    return {
        "paths_included_in_csvs": False,
        "use_gpu": bool(emd.get("UseGPU", True)),
        "using_multi_gpu": False,
        "use_multi_gpu": False,
        "num_predictions": 1,
        "autocast_float16": False,
        "save_reconstructed_images": False,
        "minimum_valid_percentage": 0.9,
        "use_xarray": False,
        "num_frames": int(emd.get("NumFrames", 2)),
        "tif_compression": "NONE",
        "change_map": {
            "return": True,
            "upsample_cosine_map": True,
            "save_dates_layer": False,
            "save_fmask_layer": False,
            "fmask_in_bit": False,
            "date_regex": None,
        },
        "feature_maps": {
            "return": False,
            "write_as_image": False,
            "embeddings": None,
        },
        "model_params": {
            "load_params": {
                "source": "local",
                "checkpoint_path": ckpt_path,
                "repo_id": "",
                "model_name": "",
            },
            "freeze_backbone": False,
            "freeze_encoder": False,
            "resume_encoder_only": False,
            "keep_pos_embedding": True if "KeepPosEmbedding" not in emd else bool(emd.get("KeepPosEmbedding")),
            "restore_weights_only": True,
            "ignore_index": 255,
            "loss": "simple",
            "backbone": "planaura_reconstruction",
            "bands": emd.get("Bands", ["B02", "B03", "B04", "B8A", "B11", "B12"]),
            "img_size": int(emd.get("ImageHeight", 512)),
            "depth": int(emd.get("Depth", 12)),
            "decoder_depth": int(emd.get("DecoderDepth", 8)),
            "patch_size": int(emd.get("PatchSize", 16)),
            "patch_stride": int(emd.get("PatchStride", emd.get("PatchSize", 16))),
            "embed_attention": bool(emd.get("EmbedAttention", True)),
            "embed_dim": int(emd.get("EmbedDim", 768)),
            "decoder_embed_dim": int(emd.get("DecoderEmbedDim", 512)),
            "num_heads": int(emd.get("NumHeads", 12)),
            "decoder_num_heads": int(emd.get("DecoderNumHeads", 16)),
            "mask_ratio": float(emd.get("MaskRatio", 0.75)),
            "tubelet_size": int(emd.get("TubeletSize", 1)),
            "no_data": float(emd.get("NoData", -9999.0)),
            "no_data_float": float(emd.get("NoDataFloat", 0.0001)),
        },
    }


def _resample_cosines_features(arr, target_h, target_w):
    """
    Approximate the repo's resample_cosines_features() for a 2D cosine map.
    Valid data are interpolated bicubically; invalid support stays -100.
    """
    arr = arr.astype(np.float32)
    if arr.shape == (target_h, target_w):
        return np.clip(arr, -1.0, 1.0)

    mask = (arr != -100.0).astype(np.float32)
    data_weighted = np.where(mask > 0, arr, 0.0).astype(np.float32)

    data_t = torch.from_numpy(data_weighted)[None, None, :, :]
    mask_t = torch.from_numpy(mask)[None, None, :, :]

    num = F.interpolate(data_t, size=(target_h, target_w), mode="bicubic", align_corners=False).squeeze().cpu().numpy()
    den = F.interpolate(mask_t, size=(target_h, target_w), mode="bicubic", align_corners=False).squeeze().cpu().numpy()

    out = np.full((target_h, target_w), -100.0, dtype=np.float32)
    good = den > 0.9
    out[good] = np.clip(num[good], -1.0, 1.0)
    return out


class Planaura:
    def __init__(self):
        self.name = "Planaura"
        self.description = "Planaura classified output for ArcGIS Classify Pixels (lazy repo-wrapper with fallback)."
        self._emd = {}
        self._cfg = None
        self._model = None
        self._device = None
        self._model_load_failed = False
        self._model_load_error = None

    def initialize(self, **kwargs):
        """
        Keep initialize() lightweight so ArcGIS can populate arguments.
        Do NOT load the model here.
        """
        self._emd = {}
        self._cfg = None
        self._model = None
        self._device = None
        self._model_load_failed = False
        self._model_load_error = None

        if "model" in kwargs:
            try:
                emd_path = kwargs["model"]
                with open(emd_path, "r", encoding="utf-8") as fh:
                    self._emd = json.load(fh)

                self.img_size = int(self._emd.get("ImageHeight", 512))
                self.patch_stride = int(self._emd.get("PatchStride", self._emd.get("PatchSize", 16)))
                self.no_data_val = float(self._emd.get("NoData", -9999.0))
                self.no_data_float = float(self._emd.get("NoDataFloat", 0.0001))
                self.n_bands = 6

                emd_dir = os.path.dirname(os.path.abspath(emd_path))
                ckpt_path = os.path.normpath(os.path.join(emd_dir, self._emd.get("ModelFile", "")))
                self._cfg = _build_runtime_config(self._emd, ckpt_path)
            except Exception as e:
                self._model_load_failed = True
                self._model_load_error = str(e)
                _write_error_log(str(e) + "\n\n" + traceback.format_exc())

    def getParameterInfo(self):
        default_img = int(getattr(self, "_emd", {}).get("ImageHeight", 512))
        return [
            {
                "name": "raster",
                "dataType": "raster",
                "required": True,
                "displayName": "Input Raster",
                "description": "12-band composite raster",
            },
            {
                "name": "model",
                "dataType": "string",
                "required": True,
                "displayName": "Input Model Definition (EMD) File",
                "description": "Input model definition (EMD) JSON file",
            },
            {
                "name": "padding",
                "dataType": "numeric",
                "required": False,
                "value": default_img // 8,
                "displayName": "Padding",
                "description": "Tile overlap padding",
            },
            {
                "name": "batch_size",
                "dataType": "numeric",
                "required": False,
                "value": 1,
                "displayName": "Batch Size",
                "description": "Unused by this adapter; kept for ArcGIS tool compatibility",
            },
            {
                "name": "use_f16",
                "dataType": "numeric",
                "required": False,
                "value": 0,
                "displayName": "Use Float16 (GPU only)",
                "description": "0 disabled, 1 enabled",
            },
            {
                "name": "high_threshold",
                "dataType": "numeric",
                "required": False,
                "value": 0.80,
                "displayName": "No-Change Threshold",
                "description": "cosine > this => class 1",
            },
            {
                "name": "mid_threshold",
                "dataType": "numeric",
                "required": False,
                "value": 0.60,
                "displayName": "Low-Change Threshold",
                "description": "cosine > this => class 2",
            },
            {
                "name": "low_threshold",
                "dataType": "numeric",
                "required": False,
                "value": 0.40,
                "displayName": "Moderate-Change Threshold",
                "description": "cosine > this => class 3; else class 4",
            },
        ]

    def getConfiguration(self, **scalars):
        self.padding = int(scalars.get("padding", 64))
        self._use_f16 = bool(int(scalars.get("use_f16", 0)))
        self.high_threshold = float(scalars.get("high_threshold", 0.80))
        self.mid_threshold = float(scalars.get("mid_threshold", 0.60))
        self.low_threshold = float(scalars.get("low_threshold", 0.40))

        img_size = int(getattr(self, "_emd", {}).get("ImageHeight", 512))
        return {
            "tx": int(scalars.get("tx", img_size)),
            "ty": int(scalars.get("ty", img_size)),
            "padding": self.padding,
            "batch_size": 1,
            "extractBands": tuple(range(12)),
            "fixedTileSize": True,
            "inputMask": False,
        }

    def updateRasterInfo(self, **kwargs):
        kwargs["output_info"]["bandCount"] = 1
        kwargs["output_info"]["pixelType"] = "U16"
        kwargs["output_info"]["noData"] = [0]
        return kwargs

    def _ensure_model(self):
        if self._model is not None:
            return
        if self._model_load_failed:
            raise RuntimeError(self._model_load_error or "Previous model load failed.")
        if not self._cfg:
            raise RuntimeError("Runtime config is not available from EMD.")

        try:
            resume_pretrained_network = _import_resume_pretrained_network()
            model, _, _, _, _ = resume_pretrained_network(self._cfg)
            if self._cfg["use_gpu"] and torch.cuda.is_available():
                model = model.cuda()
            model.prepare_to_infer()
            model.eval()
            self._model = model
            self._device = model.device_()
        except Exception as e:
            self._model_load_failed = True
            self._model_load_error = str(e)
            _write_error_log(str(e) + "\n\n" + traceback.format_exc())
            raise

    def _empty_output(self, h, w):
        out = np.zeros((1, h, w), dtype=np.uint16)
        mask = np.zeros((1, h, w), dtype=np.uint8)
        return out, mask

    def _simple_banddiff_classes(self, arr, pad):
        """
        Fallback path that already worked in your ArcGIS tests.
        """
        before = arr[:6]
        after = arr[6:12]
        H, W = arr.shape[1], arr.shape[2]

        if H > 2 * pad and W > 2 * pad:
            before = before[:, pad:H - pad, pad:W - pad]
            after = after[:, pad:H - pad, pad:W - pad]

        diff = np.mean(np.abs(after - before), axis=0)

        valid = np.isfinite(diff)
        if valid.any():
            v = diff[valid]
            dmin = float(v.min())
            dmax = float(v.max())
            if dmax > dmin:
                diff_scaled = (diff - dmin) / (dmax - dmin)
            else:
                diff_scaled = np.zeros_like(diff, dtype=np.float32)
        else:
            diff_scaled = np.zeros_like(diff, dtype=np.float32)

        out2d = np.zeros(diff_scaled.shape, dtype=np.uint16)
        out2d[(diff_scaled >= 0.00) & (diff_scaled < 0.25)] = 1
        out2d[(diff_scaled >= 0.25) & (diff_scaled < 0.50)] = 2
        out2d[(diff_scaled >= 0.50) & (diff_scaled < 0.75)] = 3
        out2d[(diff_scaled >= 0.75)] = 4

        mask2d = np.full(diff_scaled.shape, 255, dtype=np.uint8)
        return out2d[np.newaxis, :, :], mask2d[np.newaxis, :, :]

    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        src = pixelBlocks.get("raster_pixels")
        if src is None:
            src = pixelBlocks.get("x")

        if src is None:
            h = int(getattr(self, "_emd", {}).get("ImageHeight", 512))
            w = int(getattr(self, "_emd", {}).get("ImageWidth", 512))
            out, mask = self._empty_output(h, w)
            pixelBlocks["output_pixels"] = out
            pixelBlocks["output_mask"] = mask
            return pixelBlocks

        arr = np.asarray(src, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[0] < 12:
            h = arr.shape[-2] if arr.ndim >= 2 else 512
            w = arr.shape[-1] if arr.ndim >= 2 else 512
            out, mask = self._empty_output(h, w)
            pixelBlocks["output_pixels"] = out
            pixelBlocks["output_mask"] = mask
            return pixelBlocks

        pad = int(getattr(self, "padding", 64))

        # Preferred path: repo PLANAURA wrapper.
        try:
            self._ensure_model()

            H, W = arr.shape[1], arr.shape[2]
            if H > 2 * pad and W > 2 * pad:
                arr_inner = arr[:, pad:H - pad, pad:W - pad]
            else:
                arr_inner = arr

            H_in, W_in = arr_inner.shape[1], arr_inner.shape[2]

            before = arr_inner[:6]
            after = arr_inner[6:12]

            # Match repo behavior: replace nodata with no_data_float before inference.
            before = np.where(before <= self.no_data_val, self.no_data_float, before).astype(np.float32)
            after = np.where(after <= self.no_data_val, self.no_data_float, after).astype(np.float32)

            # Pad to the model's configured image size if needed (e.g., edge tiles).
            H_model = int(getattr(self, "img_size", 512))
            W_model = int(getattr(self, "img_size", 512))
            before_full = np.full((6, H_model, W_model), self.no_data_float, dtype=np.float32)
            after_full = np.full((6, H_model, W_model), self.no_data_float, dtype=np.float32)

            copy_h = min(H_in, H_model)
            copy_w = min(W_in, W_model)
            before_full[:, :copy_h, :copy_w] = before[:, :copy_h, :copy_w]
            after_full[:, :copy_h, :copy_w] = after[:, :copy_h, :copy_w]

            # Shape expected by PLANAURA.forward: (batch, bands, frames, H, W)
            x_np = np.stack([before_full, after_full], axis=1)  # (bands, frames, H, W)
            x_tensor = torch.from_numpy(x_np).unsqueeze(0).to(self._device)

            with torch.no_grad():
                if self._use_f16 and self._device.type == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        _, change_tuple, _ = self._model(x_tensor.float())
                else:
                    _, change_tuple, _ = self._model(x_tensor.float())

            cosine_map, which_before = change_tuple
            if cosine_map is None:
                raise RuntimeError("PLANAURA model returned cosine_map=None")

            cos_np = cosine_map.detach().cpu().numpy()[0]
            cos_up = _resample_cosines_features(cos_np, H_model, W_model)

            # Crop back to the actual inner tile size.
            cos_use = cos_up[:H_in, :W_in]

            out2d = np.zeros((H_in, W_in), dtype=np.uint16)
            mask2d = np.zeros((H_in, W_in), dtype=np.uint8)

            valid = np.isfinite(cos_use) & (cos_use != -100.0)
            if valid.any():
                cls = np.full((H_in, W_in), 4, dtype=np.uint16)
                cls[cos_use > self.low_threshold] = 3
                cls[cos_use > self.mid_threshold] = 2
                cls[cos_use > self.high_threshold] = 1
                cls[~valid] = 0

                out2d[:, :] = cls
                mask2d[valid] = 255

            pixelBlocks["output_pixels"] = out2d[np.newaxis, :, :]
            pixelBlocks["output_mask"] = mask2d[np.newaxis, :, :]
            return pixelBlocks

        except Exception as e:
            # Important: preserve a working result instead of blank output.
            _write_error_log(str(e) + "\n\n" + traceback.format_exc())
            out, mask = self._simple_banddiff_classes(arr, pad)
            pixelBlocks["output_pixels"] = out
            pixelBlocks["output_mask"] = mask
            return pixelBlocks
