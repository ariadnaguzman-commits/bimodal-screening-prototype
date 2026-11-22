"""
Módulo B - Extracción de Descriptores Biofísicos Celulares (SIPaKMeD)
Extrae características morfométricas (relación N/C, circularidad, excentricidad),
densitométricas/colorimétricas (CIELAB) y de textura de cromatina (GLCM) a partir
de las máscaras segmentadas de núcleo y citoplasma.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from skimage.measure import regionprops
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt

# Asegura que el directorio raíz del proyecto esté en sys.path al ejecutar el script
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.cytology.segmentation import segment_cell


def _extract_region_geometry(mask: np.ndarray, prefix: str) -> dict[str, float]:
    """Calcula descriptores geométricos de una máscara binaria usando regionprops."""
    props = regionprops(mask.astype(np.uint8))
    if not props:
        return {
            f"{prefix}_area": 0.0,
            f"{prefix}_perimeter": 0.0,
            f"{prefix}_circularity": 0.0,
            f"{prefix}_eccentricity": 0.0,
            f"{prefix}_solidity": 0.0,
            f"{prefix}_aspect_ratio": 1.0,
        }

    reg = max(props, key=lambda r: r.area)
    area = float(reg.area)
    perimeter = float(reg.perimeter)
    circularity = (4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    circularity = float(np.clip(circularity, 0.0, 1.0))

    minor_axis = float(reg.axis_minor_length)
    major_axis = float(reg.axis_major_length)
    aspect_ratio = (major_axis / minor_axis) if minor_axis > 0 else 1.0

    return {
        f"{prefix}_area": area,
        f"{prefix}_perimeter": perimeter,
        f"{prefix}_circularity": circularity,
        f"{prefix}_eccentricity": float(reg.eccentricity),
        f"{prefix}_solidity": float(reg.solidity),
        f"{prefix}_aspect_ratio": float(aspect_ratio),
    }


def _extract_glcm_features(l_channel: np.ndarray, nuc_mask: np.ndarray) -> dict[str, float]:
    """
    Calcula descriptores de textura de cromatina (GLCM) en la región nuclear
    cuantizada a 32 niveles de gris y promediada en 4 orientaciones.
    """
    ys, xs = np.where(nuc_mask)
    if ys.size == 0:
        return {
            "glcm_contrast": 0.0,
            "glcm_dissimilarity": 0.0,
            "glcm_homogeneity": 0.0,
            "glcm_energy": 0.0,
            "glcm_ASM": 0.0,
            "glcm_correlation": 0.0,
        }

    y_min, y_max = ys.min(), ys.max() + 1
    x_min, x_max = xs.min(), xs.max() + 1
    patch = l_channel[y_min:y_max, x_min:x_max].copy()
    sub_mask = nuc_mask[y_min:y_max, x_min:x_max]

    # Rellena el exterior del núcleo con la mediana nuclear para evitar bordes artificiales
    median_val = int(np.median(patch[sub_mask]))
    patch[~sub_mask] = median_val

    if patch.shape[0] < 2 or patch.shape[1] < 2:
        patch = np.pad(patch, 1, mode="edge")

    # Cuantización a 32 niveles (0..31) para estabilidad estadística en núcleos pequeños
    levels = 32
    patch_q = (patch.astype(np.float32) / 256.0 * levels).astype(np.uint8)
    patch_q = np.clip(patch_q, 0, levels - 1)

    angles = [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0]
    glcm = graycomatrix(
        patch_q,
        distances=[1],
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )

    return {
        "glcm_contrast": float(graycoprops(glcm, "contrast").mean()),
        "glcm_dissimilarity": float(graycoprops(glcm, "dissimilarity").mean()),
        "glcm_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "glcm_energy": float(graycoprops(glcm, "energy").mean()),
        "glcm_ASM": float(graycoprops(glcm, "ASM").mean()),
        "glcm_correlation": float(graycoprops(glcm, "correlation").mean()),
    }


def extract_cell_features(image_bgr: np.ndarray) -> dict[str, float] | None:
    """Extrae el vector completo de características biofísicas de una célula."""
    nuc_mask, cell_mask = segment_cell(image_bgr)
    if not nuc_mask.any() or not cell_mask.any():
        return None

    cyto_mask = cell_mask & (~nuc_mask)
    if not cyto_mask.any():
        cyto_mask = cell_mask.copy()

    # 1. Morfometría
    nuc_geom = _extract_region_geometry(nuc_mask, "nuc")
    cell_geom = _extract_region_geometry(cell_mask, "cell")

    cyto_area = float(max(1, int((cell_mask & (~nuc_mask)).sum())))
    nc_ratio = nuc_geom["nuc_area"] / cyto_area
    nc_area_fraction = nuc_geom["nuc_area"] / max(1.0, cell_geom["cell_area"])

    features: dict[str, float] = {
        **nuc_geom,
        **cell_geom,
        "cyto_area": cyto_area,
        "nc_ratio": float(nc_ratio),
        "nc_area_fraction": float(nc_area_fraction),
    }

    # 2. Colorimetría y densidad óptica en CIELAB
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    for idx, ch_name in enumerate(["L", "a", "b"]):
        ch = lab[:, :, idx]
        nuc_vals = ch[nuc_mask]
        cyto_vals = ch[cyto_mask]

        features[f"nuc_{ch_name}_mean"] = float(nuc_vals.mean())
        features[f"nuc_{ch_name}_std"] = float(nuc_vals.std())
        features[f"cyto_{ch_name}_mean"] = float(cyto_vals.mean())
        features[f"cyto_{ch_name}_std"] = float(cyto_vals.std())

    # Contraste de hipercromasia entre citoplasma y núcleo
    features["delta_L_cyto_nuc"] = features["cyto_L_mean"] - features["nuc_L_mean"]

    # 3. Textura GLCM de la cromatina nuclear
    l_uint8 = lab[:, :, 0].astype(np.uint8)
    features.update(_extract_glcm_features(l_uint8, nuc_mask))

    return features


def build_feature_dataset(dataset_dir: Path, results_dir: Path) -> pd.DataFrame:
    """Procesa todas las imágenes recortadas de SIPaKMeD y guarda el dataset tabular."""
    cropped_images = sorted(dataset_dir.rglob("CROPPED/*.bmp"))
    if not cropped_images:
        raise FileNotFoundError(
            f"No se encontraron imágenes en {dataset_dir}. "
            "Verifica la ruta data/raw/sipakmed/."
        )

    records = []
    total = len(cropped_images)
    for idx, img_path in enumerate(cropped_images, start=1):
        class_name = img_path.parent.parent.name.replace("im_", "")
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        feats = extract_cell_features(img_bgr)
        if feats is None:
            continue

        records.append({
            "image_id": img_path.stem,
            "class": class_name,
            **feats,
        })

        if idx % 1000 == 0 or idx == total:
            print(f"Procesadas {idx}/{total} células...")

    df = pd.DataFrame(records)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "cytology_features.csv"
    df.to_csv(csv_path, index=False)

    _plot_feature_distributions(df, results_dir / "feature_distributions.png")
    return df


def _plot_feature_distributions(df: pd.DataFrame, out_path: Path) -> None:
    """Genera boxplots comparativos de los 4 biomarcadores clave por categoría celular."""
    classes = sorted(df["class"].unique())
    short_labels = [c.replace("Superficial-Intermediate", "Sup-Interm") for c in classes]

    metrics = [
        ("nc_ratio", "Relación Núcleo / Citoplasma (N/C)", (0, 1.5)),
        ("nuc_area", "Área Nuclear (píxeles)", None),
        ("delta_L_cyto_nuc", "Contraste Hipercromasia ΔL* (Cito - Nuc)", None),
        ("glcm_contrast", "Textura Cromatina (Contraste GLCM)", None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    for ax, (col, title, ylim) in zip(axes, metrics):
        data_by_class = [df.loc[df["class"] == c, col].dropna().values for c in classes]
        bp = ax.boxplot(data_by_class, tick_labels=short_labels, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.tick_params(axis="x", rotation=15)
        if ylim is not None:
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    data_dir = ROOT_DIR / "data" / "raw" / "sipakmed"
    out_dir = ROOT_DIR / "results"

    print("Extrayendo características morfométricas, CIELAB y GLCM sobre SIPaKMeD...")
    features_df = build_feature_dataset(data_dir, out_dir)

    summary_cols = ["nc_ratio", "nuc_area", "nuc_circularity", "delta_L_cyto_nuc", "glcm_contrast"]
    print("\nMediana de biomarcadores clave por categoría:")
    print(features_df.groupby("class")[summary_cols].median().round(3))
    print(f"\nDataset guardado en results/cytology_features.csv ({features_df.shape[0]} filas x {features_df.shape[1]} columnas)")