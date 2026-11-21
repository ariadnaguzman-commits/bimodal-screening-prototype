"""
Módulo B - Segmentación Celular Jerárquica y Validación Geométrica (SIPaKMeD)
Pipeline: Conversión RGB -> CIELAB, separación célula/fondo mediante contraste
cromático y luminancia, extracción intranuclear mediante Otsu local + morfología,
y validación cuantitativa con Coeficiente de Dice (DSC) e IoU.
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt


def load_ground_truth_mask(dat_path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Construye una máscara binaria a partir de los archivos de contorno .dat de SIPaKMeD."""
    mask = np.zeros(shape, dtype=np.uint8)
    if not dat_path.exists():
        return mask

    coords = np.loadtxt(dat_path, delimiter=",")
    if coords.size == 0:
        return mask
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)

    pts = np.round(coords[:, :2]).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def compute_dice_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float]:
    """Calcula el Coeficiente de Similitud de Dice (DSC) y la Intersección sobre Unión (IoU)."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    denom_dice = pred.sum() + gt.sum()

    dice = (2.0 * intersection / denom_dice) if denom_dice > 0 else 1.0
    iou = (intersection / union) if union > 0 else 1.0
    return float(dice), float(iou)


def _select_best_component(
    binary_mask: np.ndarray,
    min_area: int = 10,
    intensity_map: np.ndarray | None = None,
    prefer_dark: bool = False,
) -> np.ndarray:
    """Selecciona el componente conectado más central y relevante según área y densidad óptica."""
    labeled, num_features = ndimage.label(binary_mask)
    if num_features == 0:
        return binary_mask

    h, w = binary_mask.shape
    cy, cx = h / 2.0, w / 2.0
    best_label = 0
    best_score = -np.inf

    for i in range(1, num_features + 1):
        comp = labeled == i
        area = int(comp.sum())
        if area < min_area:
            continue

        ys, xs = np.where(comp)
        dist = np.hypot(ys.mean() - cy, xs.mean() - cx)
        centrality = 1.0 / (1.0 + 0.08 * dist)

        if prefer_dark and intensity_map is not None:
            # Favorece regiones hipercrómicas (baja luminancia L*) y compactas cerca del centro
            darkness = max(1.0, 255.0 - float(intensity_map[comp].mean()))
            score = (darkness ** 2) * np.sqrt(area) * centrality
        else:
            score = area * centrality

        if score > best_score:
            best_score = score
            best_label = i

    if best_label == 0:
        return np.zeros_like(binary_mask, dtype=bool)
    return labeled == best_label


def segment_cell(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Segmentación jerárquica en espacio CIELAB:
    1) Aísla la célula completa (núcleo + citoplasma) frente al fondo iluminado.
    2) Segmenta el núcleo exclusivamente dentro de la región celular.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch = cv2.medianBlur(lab[:, :, 0], 5)
    a_ch = cv2.medianBlur(lab[:, :, 1], 5)
    b_ch = cv2.medianBlur(lab[:, :, 2], 5)

    # --- ETAPA 1: Máscara de célula completa (célula vs. fondo) ---
    # El fondo de microscopía es brillante (L* alto) y acromático (a*, b* cercanos a 128)
    chroma = np.hypot(a_ch.astype(np.float32) - 128.0, b_ch.astype(np.float32) - 128.0)
    cell_salience = cv2.normalize(
        (255.0 - l_ch.astype(np.float32)) + 1.5 * chroma,
        None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    thresh_cell = threshold_otsu(cell_salience)
    cell_raw = cell_salience > (0.85 * thresh_cell)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    cell_clean = cv2.morphologyEx(cell_raw.astype(np.uint8), cv2.MORPH_CLOSE, kernel_med)
    cell_clean = cv2.morphologyEx(cell_clean, cv2.MORPH_OPEN, kernel_small)
    cell_clean = ndimage.binary_fill_holes(cell_clean).astype(bool)
    whole_cell_mask = _select_best_component(cell_clean, min_area=150, prefer_dark=False)

    if not whole_cell_mask.any():
        whole_cell_mask = np.ones_like(l_ch, dtype=bool)

    # --- ETAPA 2: Segmentación del núcleo dentro de la célula ---
    # La cromatina nuclear absorbe más luz (L* bajo) y tiende al azul/violeta (b* bajo)
    nuc_score = cv2.normalize(
        (255.0 - l_ch.astype(np.float32)) + 0.6 * (255.0 - b_ch.astype(np.float32)),
        None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    cell_pixels = nuc_score[whole_cell_mask]
    if cell_pixels.size > 20 and cell_pixels.max() > cell_pixels.min():
        # Otsu local restringido al interior de la célula + percentil superior para núcleos picnóticos
        otsu_local = threshold_otsu(cell_pixels)
        perc_local = np.percentile(cell_pixels, 82)
        thresh_nuc = max(otsu_local, perc_local)
        nuc_raw = (nuc_score >= thresh_nuc) & whole_cell_mask
    else:
        nuc_raw = whole_cell_mask.copy()

    nuc_clean = cv2.morphologyEx(nuc_raw.astype(np.uint8), cv2.MORPH_OPEN, kernel_small)
    nuc_clean = cv2.morphologyEx(nuc_clean, cv2.MORPH_CLOSE, kernel_small)
    nuc_clean = ndimage.binary_fill_holes(nuc_clean).astype(bool)

    # Permite núcleos picnóticos pequeños (>= 8 px) típicos de células superficiales
    nuc_mask = _select_best_component(
        nuc_clean, min_area=8, intensity_map=l_ch, prefer_dark=True
    )

    # Garantiza consistencia topológica: el núcleo siempre está contenido en la célula
    whole_cell_mask = ndimage.binary_fill_holes(whole_cell_mask | nuc_mask)
    return nuc_mask, whole_cell_mask


def evaluate_dataset(dataset_dir: Path, results_dir: Path, max_samples_per_class: int | None = None) -> pd.DataFrame:
    """Evalúa la segmentación sobre SIPaKMeD y genera figura comparativa para el reporte."""
    records = []
    class_examples = {}

    cropped_images = sorted(dataset_dir.rglob("CROPPED/*.bmp"))
    if not cropped_images:
        raise FileNotFoundError(
            f"No se encontraron imágenes en {dataset_dir}. "
            "Verifica que las carpetas de SIPaKMeD estén dentro de data/raw/sipakmed/."
        )

    for img_path in cropped_images:
        class_name = img_path.parent.parent.name.replace("im_", "")
        if max_samples_per_class is not None:
            current_count = sum(1 for r in records if r["class"] == class_name)
            if current_count >= max_samples_per_class:
                continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        h, w = img_bgr.shape[:2]
        nuc_dat = img_path.with_name(f"{img_path.stem}_nuc.dat")
        cyt_dat = img_path.with_name(f"{img_path.stem}_cyt.dat")

        gt_nuc = load_ground_truth_mask(nuc_dat, (h, w))
        gt_cyt = load_ground_truth_mask(cyt_dat, (h, w)) | gt_nuc

        pred_nuc, pred_cell = segment_cell(img_bgr)

        dice_nuc, iou_nuc = compute_dice_iou(pred_nuc, gt_nuc)
        dice_cell, iou_cell = compute_dice_iou(pred_cell, gt_cyt)

        records.append({
            "image_id": img_path.stem,
            "class": class_name,
            "dice_nucleus": dice_nuc,
            "iou_nucleus": iou_nuc,
            "dice_cell": dice_cell,
            "iou_cell": iou_cell,
        })

        # Guarda un ejemplo representativo bien segmentado por clase para la figura
        if gt_nuc.any() and gt_cyt.any():
            score_pair = dice_nuc + dice_cell
            if class_name not in class_examples or score_pair > class_examples[class_name][-1]:
                class_examples[class_name] = (
                    img_bgr, gt_nuc, gt_cyt, pred_nuc, pred_cell, dice_nuc, dice_cell, score_pair
                )

    df = pd.DataFrame(records)
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "segmentation_metrics.csv", index=False)

    if class_examples:
        n_classes = len(class_examples)
        fig, axes = plt.subplots(n_classes, 3, figsize=(10, 3.2 * n_classes))
        if n_classes == 1:
            axes = np.expand_dims(axes, axis=0)

        for row, (cls_name, data) in enumerate(sorted(class_examples.items())):
            img_bgr, gt_nuc, gt_cyt, pred_nuc, pred_cell, d_nuc, d_cell, _ = data
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            gt_overlay = img_rgb.copy()
            gt_overlay[gt_cyt] = (gt_overlay[gt_cyt] * 0.65 + np.array([0, 200, 100]) * 0.35).astype(np.uint8)
            gt_overlay[gt_nuc] = (gt_overlay[gt_nuc] * 0.5 + np.array([255, 50, 50]) * 0.5).astype(np.uint8)

            pred_overlay = img_rgb.copy()
            pred_overlay[pred_cell] = (pred_overlay[pred_cell] * 0.65 + np.array([0, 150, 255]) * 0.35).astype(np.uint8)
            pred_overlay[pred_nuc] = (pred_overlay[pred_nuc] * 0.5 + np.array([255, 220, 0]) * 0.5).astype(np.uint8)

            axes[row, 0].imshow(img_rgb)
            axes[row, 0].set_title(f"{cls_name}\nOriginal")
            axes[row, 1].imshow(gt_overlay)
            axes[row, 1].set_title("Ground Truth (.dat)")
            axes[row, 2].imshow(pred_overlay)
            axes[row, 2].set_title(f"Segmentación (CIELAB+Otsu)\nDSC Nuc: {d_nuc:.2f} | Cél: {d_cell:.2f}")

            for col in range(3):
                axes[row, col].axis("off")

        plt.tight_layout()
        plt.savefig(results_dir / "segmentation_examples.png", dpi=300)
        plt.close()

    return df


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data" / "raw" / "sipakmed"
    out_dir = base_dir / "results"

    print("Ejecutando validación de segmentación sobre SIPaKMeD...")
    metrics_df = evaluate_dataset(data_dir, out_dir)
    print("\nResumen por clase (promedio):")
    print(metrics_df.groupby("class")[["dice_nucleus", "iou_nucleus", "dice_cell", "iou_cell"]].mean().round(3))
    print(f"\nPromedio global DSC Núcleo: {metrics_df['dice_nucleus'].mean():.3f}")
    print(f"Promedio global DSC Célula: {metrics_df['dice_cell'].mean():.3f}")